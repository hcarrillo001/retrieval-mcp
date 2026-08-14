"""
Touchstone — an MCP eval server. DeepEval-style metrics, golden sets, and
authored custom metrics, callable from Claude Desktop, Agent Builder, or CI.

Run (stdio, for Claude Desktop):   python server.py
Configure the judge via env vars (see judge.py).

Tools
  list_metrics()                              what can I score with?
  load_golden_set(name, source, fmt)          register a golden set (file/inline)
  list_golden_sets()                          what's loaded?
  author_metric(name, criteria, examples)     turn plain language into a scorer
  run_eval(golden_set, metrics, outputs)      score a golden set's outputs
  evaluate_case(input, actual_output, ...)    one-off score without a golden set
"""
from __future__ import annotations
import os
import statistics
from typing import List, Optional

from mcp.server.fastmcp import FastMCP, Image

import metrics as M
import history as H
import charts as C
from judge import judge_json, budget_status, reset_spend, BudgetExceeded
from goldensets import load_records

mcp = FastMCP("retrieval-mcp")

# in-memory state for the session
GOLDEN_SETS: dict[str, list] = {}
CUSTOM_METRICS: dict[str, list] = {}  # name -> evaluation_steps


def _run_metric(name: str, case: dict, threshold: float) -> dict:
    if name in M.BUILTIN:
        return M.BUILTIN[name](case, judge_json, threshold=threshold)
    if name in CUSTOM_METRICS:
        return M.g_eval(case, judge_json, name, CUSTOM_METRICS[name], threshold=threshold)
    raise ValueError(f"Unknown metric '{name}'. Use list_metrics().")


@mcp.tool()
def list_metrics() -> dict:
    """List built-in metrics and any custom metrics authored this session."""
    return {
        "builtin": sorted(M.BUILTIN.keys()),
        "custom": sorted(CUSTOM_METRICS.keys()),
        "notes": "faithfulness/contextual_* need retrieval_context; "
                 "hallucination needs context; answer_relevancy needs input+actual_output.",
    }


@mcp.tool()
def load_golden_set(name: str, source: str, fmt: str = "auto") -> dict:
    """Load a golden set from a file path, a JSON array string, or JSONL text.
    Field names are normalized (question->input, answer->actual_output,
    ground_truth->expected_output, contexts->context, passages->retrieval_context, ...)."""
    records = load_records(source, fmt)
    GOLDEN_SETS[name] = records
    keys = sorted({k for r in records for k in r})
    return {"name": name, "cases": len(records), "detected_fields": keys,
            "preview": records[0] if records else {}}


@mcp.tool()
def list_golden_sets() -> dict:
    """Show loaded golden sets and their sizes."""
    return {n: len(c) for n, c in GOLDEN_SETS.items()}


@mcp.tool()
def author_metric(name: str, criteria: str, examples: Optional[List[str]] = None) -> dict:
    """Author a custom metric from a plain-language description (+ optional golden
    examples). The judge converts your criteria into explicit evaluation steps,
    which run as a G-Eval-style scorer in run_eval / evaluate_case."""
    steps = M.author_evaluation_steps(criteria, examples or [], judge_json)
    CUSTOM_METRICS[name] = steps
    return {"metric": name, "evaluation_steps": steps,
            "usage": f"pass metrics=['{name}'] to run_eval / evaluate_case"}


@mcp.tool()
def run_eval(golden_set: str, metrics: List[str], threshold: float = 0.7,
             outputs: Optional[List[str]] = None, label: str = "",
             limit: int = 3, generator_model: str = "", judge_model: str = "") -> dict:
    """Run metrics over a loaded golden set. All cases are scored, but only the
    `limit` lowest-scoring cases are returned by default (3) to keep replies short;
    `total_cases`/`shown` tell you how many more exist — call show_run_cases to page.
    Pass `generator_model` (which LLM produced the outputs) and `judge_model` (which
    LLM scored them) so the dashboard can compare across models; judge_model defaults
    to the configured judge. The run is saved to history (file or Supabase).
    Raises if the spend cap is hit mid-run; partial spend is still metered."""
    if golden_set not in GOLDEN_SETS:
        raise ValueError(f"No golden set '{golden_set}'. Load one first.")
    judge_model = judge_model or os.environ.get("RETRIEVAL_JUDGE_MODEL", "claude-sonnet-4-6")
    cases = GOLDEN_SETS[golden_set]
    if outputs is not None:
        if len(outputs) != len(cases):
            raise ValueError(f"{len(outputs)} outputs vs {len(cases)} cases")
        cases = [{**c, "actual_output": o} for c, o in zip(cases, outputs)]

    full, scores = [], {m: [] for m in metrics}
    try:
        for i, case in enumerate(cases):
            row = {"index": i, "input": case.get("input", ""),
                   "actual_output": case.get("actual_output", ""),
                   "expected_output": case.get("expected_output", ""),
                   "retrieval_context": case.get("retrieval_context") or case.get("context") or [],
                   "scores": {}, "min_score": 1.0}
            for m in metrics:
                res = _run_metric(m, case, threshold)
                row["scores"][m] = {"score": res["score"], "success": res["success"],
                                    "reason": res["reason"], "details": res["details"]}
                scores[m].append(res["score"])
                row["min_score"] = min(row["min_score"], res["score"])
            full.append(row)
    except BudgetExceeded as e:
        return {"error": "budget_exceeded", "message": str(e),
                "cases_scored": len(full), "budget": budget_status()}

    aggregate = {
        m: {"mean_score": round(statistics.mean(s), 4),
            "pass_rate": round(sum(x >= threshold for x in s) / len(s), 4),
            "n": len(s)}
        for m, s in scores.items() if s
    }
    run_id = H.save_run(golden_set, threshold, aggregate, label, per_case=full,
                        generator_model=generator_model, judge_model=judge_model)

    worst = sorted(full, key=lambda r: r["min_score"])[:limit]
    return {"run_id": run_id, "golden_set": golden_set, "threshold": threshold,
            "generator_model": generator_model, "judge_model": judge_model,
            "aggregate": aggregate, "budget": budget_status(),
            "total_cases": len(full), "shown": len(worst),
            "showing": "lowest-scoring first" if len(full) > len(worst) else "all",
            "per_case": worst,
            "more": (f"{len(full) - len(worst)} more — call show_run_cases('{run_id}')"
                     if len(full) > len(worst) else None)}


@mcp.tool()
def show_run_cases(run_id: str = "latest", offset: int = 0, limit: int = 10,
                   metric: str = "") -> dict:
    """Page through the full per-case results of a saved run (the cases run_eval
    didn't show). Optionally sort by a single metric's score (worst first)."""
    run = H.get_run(run_id)
    if not run:
        raise ValueError(f"No run '{run_id}'.")
    rows = run.get("per_case", [])
    if metric:
        rows = sorted(rows, key=lambda r: r["scores"].get(metric, {}).get("score", 1.0))
    window = rows[offset:offset + limit]
    return {"run_id": run["run_id"], "total_cases": len(rows),
            "offset": offset, "returned": len(window), "per_case": window}


@mcp.tool()
def get_budget() -> dict:
    """Current judge spend, the cap (RETRIEVAL_BUDGET_USD), and remaining headroom."""
    return budget_status()


@mcp.tool()
def reset_budget() -> dict:
    """Reset the cumulative spend ledger to $0 (the cap itself is unchanged)."""
    reset_spend()
    return budget_status()


@mcp.tool()
def evaluate_case(input: str, actual_output: str, metrics: List[str],
                  expected_output: str = "", context: Optional[List[str]] = None,
                  retrieval_context: Optional[List[str]] = None,
                  threshold: float = 0.7) -> dict:
    """Score a single output inline, without loading a golden set."""
    case = {"input": input, "actual_output": actual_output,
            "expected_output": expected_output,
            "context": context or [], "retrieval_context": retrieval_context or []}
    return {m: _run_metric(m, case, threshold) for m in metrics}


@mcp.tool()
def list_runs(golden_set: str = "", last_n: int = 20) -> dict:
    """List saved runs (most recent last) with their per-metric mean scores."""
    runs = H.load_runs(golden_set or None, last_n)
    return {"runs": [
        {"run_id": r["run_id"], "label": r["label"], "timestamp": r["timestamp"],
         "golden_set": r["golden_set"],
         "means": {m: a["mean_score"] for m, a in r["aggregate"].items()}}
        for r in runs]}


@mcp.tool()
def plot_metric_trend(metric: str, golden_set: str = "", last_n: int = 10,
                      threshold: float = 0.7, show_range: bool = True) -> Image:
    """Line chart of one metric's mean score across recent runs, with a threshold
    line and (optionally) a shaded min-max range band for the window."""
    runs = H.load_runs(golden_set or None, last_n)
    if not runs:
        raise ValueError("No runs saved yet. Run an eval first.")
    return Image(data=C.trend(runs, metric, threshold, show_range), format="png")


@mcp.tool()
def plot_run(run_id: str = "latest", golden_set: str = "",
             threshold: float = 0.7) -> Image:
    """Bar chart of every metric's mean score for a single run (default: latest).
    Bars are green/red by pass/fail against the threshold."""
    if run_id == "latest":
        runs = H.load_runs(golden_set or None, last_n=1)
        if not runs:
            raise ValueError("No runs saved yet.")
        run = runs[-1]
    else:
        run = H.get_run(run_id)
        if not run:
            raise ValueError(f"No run '{run_id}'.")
    return Image(data=C.run_bars(run, threshold), format="png")


@mcp.tool()
def compare_runs(run_ids: Optional[List[str]] = None, golden_set: str = "",
                 last_n: int = 2, threshold: float = 0.7) -> Image:
    """Grouped-bar comparison of several runs across all shared metrics.
    Pass explicit run_ids, or omit to compare the last_n runs."""
    if run_ids:
        runs = [H.get_run(r) for r in run_ids]
        runs = [r for r in runs if r]
    else:
        runs = H.load_runs(golden_set or None, last_n)
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare.")
    return Image(data=C.compare(runs, threshold), format="png")


@mcp.tool()
def ground_against_url(url: str, output: str, question: str = "",
                       threshold: float = 0.7) -> dict:
    """Fetch a web page and check OUTPUT for *consistency* with it: faithfulness
    (every claim supported by the page) and, if a question is given, answer
    relevancy. Use it to ask "does my answer agree with what this page says?".

    IMPORTANT — this is NOT a correctness check. There are no labels here: the
    page is an unverified source, not ground truth. A high score means the output
    matches the page, not that either is right. The returned `caveat` says so on
    every call; surface it to the user."""
    from web import fetch_text

    page = fetch_text(url)
    case = {"input": question, "actual_output": output,
            "retrieval_context": [page], "context": [page]}
    result = {"url": url, "chars_used": len(page),
              "faithfulness": _run_metric("faithfulness", case, threshold),
              "budget": budget_status()}
    if question:
        result["answer_relevancy"] = _run_metric("answer_relevancy", case, threshold)
    result["caveat"] = (
        "Consistency only, not correctness: this scores agreement with an "
        "unverified web page, with no labels (expected answer / known-relevant "
        "context). A high score means 'matches the page' — the page itself may "
        "be wrong, biased, or out of date."
    )
    return result


if __name__ == "__main__":
    mcp.run()
