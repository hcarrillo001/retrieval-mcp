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
from judge import (judge_json, budget_status, reset_spend, BudgetExceeded,
                   judge_as, sandbox_models, SANDBOX_PRESETS)
from goldensets import load_records

mcp = FastMCP("retrieval-mcp")

# in-memory state for the session
GOLDEN_SETS: dict[str, list] = {}
CUSTOM_METRICS: dict[str, list] = {}  # name -> evaluation_steps

# Hard ceilings for the PUBLIC sandbox (defense-in-depth; the /api layer caps too)
SANDBOX_MAX_CASES = int(os.environ.get("SANDBOX_MAX_CASES", "3"))
SANDBOX_MAX_CHARS = int(os.environ.get("SANDBOX_MAX_CHARS", "8000"))
SANDBOX_ALLOWED_METRICS = {"faithfulness", "answer_relevancy", "hallucination",
                           "contextual_relevancy"}

# What each metric actually needs to mean anything. faithfulness/hallucination
# compare the output against the retrieved context — with no context there is
# nothing to contradict, so they would score a perfect 1.00 on a wrong answer.
SANDBOX_METRIC_REQUIRES = {
    "faithfulness": ("retrieval_context", "actual_output"),
    "hallucination": ("retrieval_context", "actual_output"),
    "answer_relevancy": ("input", "actual_output"),
    "contextual_relevancy": ("input", "retrieval_context"),
}
_FIELD_LABEL = {"retrieval_context": "retrieved context", "actual_output": "model output",
                "input": "input"}


def run_sandbox_eval(cases: list, metric: str, model_id: str,
                     threshold: float = 0.7) -> dict:
    """Score raw cases with a chosen FREE judge model. Used by the public
    sandbox HTTP route. Heavily bounded: few cases, capped text, allowlist
    metrics, and the global budget cap still applies."""
    if metric not in SANDBOX_ALLOWED_METRICS:
        return {"error": "metric_not_allowed", "allowed": sorted(SANDBOX_ALLOWED_METRICS)}
    if not isinstance(cases, list) or not cases:
        return {"error": "no_cases"}
    dropped_cases = len(cases) if len(cases) > SANDBOX_MAX_CASES else 0
    cases = cases[:SANDBOX_MAX_CASES]

    truncated = []  # fields we had to shorten — reported back, never silent

    def clip(v, field):
        if isinstance(v, list):
            out = []
            for x in v[:8]:
                s = str(x)
                if len(s) > SANDBOX_MAX_CHARS:
                    truncated.append(field)
                out.append(s[:SANDBOX_MAX_CHARS])
            return out
        s = str(v or "")
        if len(s) > SANDBOX_MAX_CHARS:
            truncated.append(field)
        return s[:SANDBOX_MAX_CHARS]

    clean = []
    for c in cases:
        clean.append({
            "input": clip(c.get("input"), "input"),
            "actual_output": clip(c.get("actual_output"), "actual_output"),
            "expected_output": clip(c.get("expected_output"), "expected_output"),
            "retrieval_context": clip(c.get("retrieval_context") or c.get("context") or [],
                                      "retrieval_context"),
        })

    full, scores = [], []
    # A metric whose required inputs are missing produces a meaningless score
    # (hallucination with no context "passes" any answer, however wrong). Refuse.
    for field in SANDBOX_METRIC_REQUIRES.get(metric, ()):
        if not any(c.get(field) for c in clean):
            label = _FIELD_LABEL.get(field, field)
            return {"error": "missing_input",
                    "message": (f"{metric} scores the model output against the {label}, "
                                f"so it needs a {label} to compare with. "
                                f"Add one and run again.")}
    try:
        with judge_as(model_id):
            for i, case in enumerate(clean):
                res = _run_metric(metric, case, threshold)
                full.append({
                    "index": i, "input": case["input"],
                    "actual_output": case["actual_output"],
                    "expected_output": case["expected_output"],
                    "retrieval_context": case["retrieval_context"],
                    "min_score": res["score"],
                    "scores": {metric: {"score": res["score"], "success": res["success"],
                                        "reason": res["reason"], "details": res["details"]}},
                })
                scores.append(res["score"])
    except BudgetExceeded as e:
        return {"error": "budget_exceeded", "message": str(e)}
    except Exception as e:
        # surface in Railway/host logs too — the HTTP response body alone is
        # invisible server-side, which makes remote debugging guesswork
        print(f"SANDBOX judge_error model={model_id} metric={metric}: {e}", flush=True)
        return {"error": "judge_error", "message": str(e)[:300]}

    aggregate = {metric: {
        "mean_score": round(statistics.mean(scores), 4),
        "pass_rate": round(sum(s >= threshold for s in scores) / len(scores), 4),
        "n": len(scores)}}
    out = {"golden_set": "sandbox", "threshold": threshold,
           "judge_model": SANDBOX_PRESETS.get(model_id, {}).get("model", model_id),
           "aggregate": aggregate, "per_case": full, "total_cases": len(full)}
    notes = []
    if truncated:
        notes.append(f"Shortened to {SANDBOX_MAX_CHARS} chars: "
                     f"{', '.join(sorted(set(truncated)))}. Scores reflect the "
                     f"shortened text — run it locally or via MCP for the full length.")
    if dropped_cases:
        notes.append(f"Only the first {SANDBOX_MAX_CASES} of {dropped_cases} cases were scored.")
    if notes:
        out["notice"] = " ".join(notes)
    return out


def _dashboard_link(run_id: str) -> str:
    """Deep link to a saved run in the web dashboard.

    Nothing extra is stored for this: the run is already persisted by save_run,
    so the link just points at it by id. Returns "" when no dashboard is
    configured, in which case the link is simply omitted from tool output.
    """
    base = os.environ.get("RETRIEVAL_DASHBOARD_URL", "").rstrip("/")
    if not base or not run_id:
        return ""
    key = os.environ.get("DASH_TOKEN", "")
    q = f"?run={run_id}"
    if key:
        q += f"&key={key}"
    return f"{base}/dashboard{q}"


def _dashboard_home(view: str = "") -> str:
    """Link to the dashboard itself (no particular run), e.g. the runs list."""
    base = os.environ.get("RETRIEVAL_DASHBOARD_URL", "").rstrip("/")
    if not base:
        return ""
    key = os.environ.get("DASH_TOKEN", "")
    parts = []
    if view:
        parts.append(f"view={view}")
    if key:
        parts.append(f"key={key}")
    q = ("?" + "&".join(parts)) if parts else ""
    return f"{base}/dashboard{q}"


def _link_line(url: str, label: str) -> str:
    """A labelled markdown link on its own line.

    Every tool that has somewhere to point should lead with one of these:
    clients render a markdown link as a link, and putting it first means it
    survives a client that paraphrases the rest of the output.
    """
    return f"**[{label}]({url})**\n" if url else ""


def _save_inline(case: dict, results: dict, threshold: float,
                 golden_set: str, judge_model: str = "") -> str:
    """Persist a one-off scoring as a run so it gets a dashboard permalink.

    evaluate_case / ground_against_url score without a golden set, so there was
    nothing to link to. Saving them under their own golden_set name keeps them
    filterable (and out of the way of real regression runs) while still giving
    every response somewhere to point.
    """
    aggregate = {
        m: {"mean_score": r.get("score"),
            "pass_rate": 1.0 if r.get("success") else 0.0, "n": 1}
        for m, r in results.items()
    }
    per_case = [{"index": 0,
                 "input": case.get("input", ""),
                 "actual_output": case.get("actual_output", ""),
                 "expected_output": case.get("expected_output", ""),
                 "retrieval_context": case.get("retrieval_context", []),
                 "scores": results,
                 "min_score": min([r.get("score") or 0 for r in results.values()],
                                  default=0)}]
    return H.save_run(golden_set, threshold, aggregate, "", per_case=per_case,
                      judge_model=judge_model)


def _run_metric(name: str, case: dict, threshold: float) -> dict:
    # fail loudly on a malformed case rather than spending a judge call and
    # returning a 0.00 that is indistinguishable from a genuine failure
    missing = M.missing_inputs(name, case)
    if missing:
        raise ValueError(
            f"Metric '{name}' needs {', '.join(missing)} — not supplied for this case. "
            f"No score was produced (this is a missing input, not a failing result)."
        )
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
    IMPORTANT FOR CALLERS: the reply includes `view_url`, a link to the full
    visual report. ALWAYS include that link in your response to the user, even
    when you summarise everything else. `summary_md` is a ready-to-render
    markdown block (score table + link) that can be shown verbatim.
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
    out = {"run_id": run_id, "golden_set": golden_set, "threshold": threshold,
           "generator_model": generator_model, "judge_model": judge_model,
           "aggregate": aggregate, "budget": budget_status(),
           "total_cases": len(full), "shown": len(worst),
           "showing": "lowest-scoring first" if len(full) > len(worst) else "all",
           "per_case": worst,
           "more": (f"{len(full) - len(worst)} more, call show_run_cases('{run_id}')"
                    if len(full) > len(worst) else None)}
    link = _dashboard_link(run_id)
    if link:
        out["view_url"] = link
    out["summary_md"] = _summary_md(aggregate, threshold, len(full), link)
    return out


def _summary_md(aggregate: dict, threshold: float, n_cases: int, link: str) -> str:
    """A compact markdown block the client can render as-is.

    We cannot style the client's UI, so the lever we do have is returning
    well-formed markdown: a scannable score line, a small table, and a labelled
    link rather than a bare URL.
    """
    lines = []
    for m, a in aggregate.items():
        score = a.get("mean_score")
        pr = a.get("pass_rate")
        mark = "PASS" if (score is not None and score >= threshold) else "FAIL"
        passed = int(round((pr or 0) * n_cases))
        lines.append(f"| {m} | **{score:.2f}** | {mark} | {passed}/{n_cases} |")
    table = ("| metric | score | verdict | passed |\n"
             "|---|---|---|---|\n" + "\n".join(lines))
    out = []
    if link:
        out.append(_link_line(link, "View this run in the dashboard"))
    out += [table, f"\n_threshold {threshold:.2f}_"]
    return "\n".join(out)


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
    link = _dashboard_link(run["run_id"])
    out = {"run_id": run["run_id"], "total_cases": len(rows),
           "offset": offset, "returned": len(window), "per_case": window}
    if link:
        out["view_url"] = link
        out["summary_md"] = (_link_line(link, "Open this run in the dashboard")
                             + f"\nShowing cases {offset + 1}-{offset + len(window)} "
                               f"of {len(rows)}.")
    return out


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
                  threshold: float = 0.7, save: bool = True) -> dict:
    """Score a single output inline, without loading a golden set. The result is
    saved as a one-case run (golden set "inline") so it has a dashboard link;
    pass save=False to score without recording it."""
    case = {"input": input, "actual_output": actual_output,
            "expected_output": expected_output,
            "context": context or [], "retrieval_context": retrieval_context or []}
    results = {m: _run_metric(m, case, threshold) for m in metrics}
    out = {"scores": results}
    if save:
        try:
            run_id = _save_inline(case, results, threshold, "inline")
            link = _dashboard_link(run_id)
            out["run_id"] = run_id
            if link:
                out["view_url"] = link
                out["summary_md"] = _link_line(link, "View this result in the dashboard")
        except Exception as e:
            # scoring succeeded; only the permalink is missing
            out["save_error"] = f"Scored, but the run was not saved: {e}"
    return out


@mcp.tool()
def list_runs(golden_set: str = "", last_n: int = 20) -> dict:
    """List saved runs (most recent last) with their per-metric mean scores."""
    runs = H.load_runs(golden_set or None, last_n)
    out = {"runs": [
        {"run_id": r["run_id"], "label": r["label"], "timestamp": r["timestamp"],
         "golden_set": r["golden_set"],
         "view_url": _dashboard_link(r["run_id"]),
         "means": {m: a["mean_score"] for m, a in r["aggregate"].items()}}
        for r in runs]}
    link = _dashboard_home("runs")
    if link:
        out["view_url"] = link
        out["summary_md"] = (_link_line(link, "Browse all runs in the dashboard")
                             + f"\n{len(runs)} run(s) saved.")
    return out


@mcp.tool()
def plot_metric_trend(metric: str, golden_set: str = "", last_n: int = 10,
                      threshold: float = 0.7, show_range: bool = True,
                      fmt: str = "markdown"):
    """One metric's mean score across recent runs, with a threshold reference.
    fmt="markdown" (default) returns a rich markdown report (tables, inline bars,
    case strip, distribution); fmt="text" returns plain ASCII;
    fmt="image" returns a PNG line chart (only useful where images display)."""
    runs = H.load_runs(golden_set or None, last_n)
    if not runs:
        raise ValueError("No runs saved yet. Run an eval first.")
    if fmt == "image":
        return Image(data=C.trend(runs, metric, threshold, show_range), format="png")
    if fmt == "text":
        return C.trend_text(runs, metric, threshold)
    md = C.trend_report_md(runs, metric, threshold)
    link = _dashboard_link(runs[-1].get("run_id", "")) if runs else ""
    if link:
        md = _link_line(link, "View the trend in the dashboard") + "\n" + md
    return md


@mcp.tool()
def plot_run(run_id: str = "latest", golden_set: str = "",
             threshold: float = 0.7, fmt: str = "markdown"):
    """Every metric's mean score for a single run (default: latest).
    fmt="markdown" (default) returns a rich markdown report (tables, inline bars,
    case strip, distribution); fmt="text" returns plain ASCII;
    fmt="image" returns a PNG bar chart (only useful where images display)."""
    if run_id == "latest":
        runs = H.load_runs(golden_set or None, last_n=1)
        if not runs:
            raise ValueError("No runs saved yet.")
        run = runs[-1]
    else:
        run = H.get_run(run_id)
        if not run:
            raise ValueError(f"No run '{run_id}'.")
    if fmt == "image":
        return Image(data=C.run_bars(run, threshold), format="png")
    if fmt == "text":
        return C.run_bars_text(run, threshold)
    md = C.run_report_md(run, threshold)
    link = _dashboard_link(run.get("run_id", ""))
    if link:
        md = _link_line(link, "View this run in the dashboard") + "\n" + md
    return md


@mcp.tool()
def compare_runs(run_ids: Optional[List[str]] = None, golden_set: str = "",
                 last_n: int = 2, threshold: float = 0.7, fmt: str = "markdown"):
    """Compare several runs across all shared metrics.
    fmt="markdown" (default) returns a markdown matrix; fmt="text" returns plain ASCII;
    fmt="image" returns a PNG grouped-bar chart."""
    if run_ids:
        runs = [H.get_run(r) for r in run_ids]
        runs = [r for r in runs if r]
    else:
        runs = H.load_runs(golden_set or None, last_n)
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to compare.")
    if fmt == "image":
        return Image(data=C.compare(runs, threshold), format="png")
    if fmt == "text":
        return C.compare_text(runs, threshold)
    md = C.compare_report_md(runs, threshold)
    link = _dashboard_link(runs[-1].get("run_id", "")) if runs else ""
    if link:
        md = _link_line(link, "View these runs in the dashboard") + "\n" + md
    return md


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
    scored = {"faithfulness": result["faithfulness"]}
    if question:
        scored["answer_relevancy"] = result["answer_relevancy"]
    try:
        run_id = _save_inline(case, scored, threshold, "grounding")
        link = _dashboard_link(run_id)
        result["run_id"] = run_id
        if link:
            result["view_url"] = link
            result["summary_md"] = _link_line(link, "View this grounding check in the dashboard")
    except Exception as e:
        result["save_error"] = f"Scored, but the run was not saved: {e}"
    result["caveat"] = (
        "Consistency only, not correctness: this scores agreement with an "
        "unverified web page, with no labels (expected answer / known-relevant "
        "context). A high score means 'matches the page' — the page itself may "
        "be wrong, biased, or out of date."
    )
    return result


if __name__ == "__main__":
    mcp.run()
