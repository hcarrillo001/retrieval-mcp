"""The run verdict (why a run passed or failed), report links, and the widget.

    cd retrieval-mcp && python -m pytest tests/ -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics as M  # noqa: E402


def _row(i, q, score, reason, details=None, metric="faithfulness", thr=0.7):
    return {"index": i, "input": q,
            "scores": {metric: {"score": score, "success": score >= thr,
                                "reason": reason, "details": details or {}}}}


def test_explain_multi_case_names_the_failing_cases():
    rows = [_row(0, "When was it completed?", 1.0, "grounded"),
            _row(1, "What is it made of?", 0.0, "Claims solid gold; context says iron."),
            _row(2, "How many moons?", 0.0, "Says 12 moons; context says two.")]
    v = M.explain("faithfulness", rows, 0.7)
    assert v["status"] == "fail"
    assert v["headline"].startswith("Fails: faithfulness averaged 0.33, below the 0.70 threshold")
    assert "(1 of 3 cases pass)" in v["headline"]
    assert len(v["drivers"]) == 2 and "solid gold" in v["drivers"][0]


def test_explain_single_case_lists_the_bad_claims():
    claims = [{"claim": "Fare booked on the airline site is expensed.", "p": 0.04, "bad": True},
              {"claim": "Economy class.", "p": 0.93, "bad": False},
              {"claim": "Booked in the portal.", "p": 0.95, "bad": False}]
    v = M.explain("faithfulness", [_row(0, "q", 0.67, "r", {"claims": claims, "kind": "support"})], 0.7)
    assert v["status"] == "fail"
    assert v["drivers"][0] == "1 of 3 claims are not supported by the context. Each one lowers the score:"
    assert "p=0.04" in v["drivers"][1]
    assert v["drivers"][-1] == "It missed the threshold by 0.03."


def test_explain_pass_and_hallucination_wording():
    claims = [{"claim": "a", "p": 0.9, "bad": False, "verdict": "supports"},
              {"claim": "b", "p": 0.9, "bad": False, "verdict": "says_nothing"}]
    v = M.explain("hallucination", [_row(0, "q", 1.0, "r", {"claims": claims, "kind": "contradiction",
                                                           "not_in_context": 1}, "hallucination")], 0.7)
    assert v["status"] == "pass" and v["headline"].startswith("Passes")
    assert any("aren't in the context" in d for d in v["drivers"])


def test_explain_llm_judge_uses_reason_and_lists():
    v = M.explain("faithfulness", [_row(0, "q", 0.0, "Invents solid gold.",
                                         {"unsupported_claims": ["made of solid gold"]})], 0.7)
    assert v["drivers"] == ["Invents solid gold.", "“made of solid gold”"]


def _call(tool, args):
    import asyncio
    import server
    from mcp.shared.memory import create_connected_server_and_client_session as conn

    async def go():
        async with conn(server.mcp._mcp_server) as c:
            return await c.call_tool(tool, args)
    return asyncio.run(go())


def test_run_eval_returns_verdict_report_url_and_summary(tmp_path, monkeypatch):
    import server
    monkeypatch.setenv("RETRIEVAL_HOME", str(tmp_path))
    monkeypatch.setenv("RETRIEVAL_DASHBOARD_URL", "https://www.retrieval-mcp.com")
    monkeypatch.delenv("DASH_TOKEN", raising=False)
    monkeypatch.delenv("RETRIEVAL_JUDGE_BACKEND", raising=False)
    monkeypatch.setattr(server, "judge_json", lambda s, u, *a: {"score": 0.4, "reasoning": "Invents a fact.",
                                                                "unsupported_claims": ["gold"]})
    cases = json.dumps([{"input": "q", "actual_output": "gold", "retrieval_context": ["iron"]}])
    sc = _call("run_eval", {"metrics": ["faithfulness"], "cases": cases}).structuredContent
    v = sc["aggregate"]["faithfulness"]["verdict"]
    assert v["status"] == "fail" and v["drivers"][0] == "Invents a fact."
    assert sc["report_url"] == f"https://www.retrieval-mcp.com/report?run={sc['run_id']}"
    md = sc["summary_md"]
    # the card: a diff block (red "-", green "+"), then the report link
    assert md.startswith("```diff\n  faithfulness 0.40 · FAIL   (threshold 0.70, 1 case)")
    assert "\n- ✕ Invents a fact." in md
    assert md.rstrip().endswith(f"(https://www.retrieval-mcp.com/report?run={sc['run_id']})**")
    # the verdict is saved with the run, so the report page has it
    import history as H
    assert H.get_run(sc["run_id"])["aggregate"]["faithfulness"]["verdict"] == v


def test_evaluate_case_has_verdict(tmp_path, monkeypatch):
    import server
    monkeypatch.setenv("RETRIEVAL_HOME", str(tmp_path))
    monkeypatch.delenv("RETRIEVAL_JUDGE_BACKEND", raising=False)
    monkeypatch.setattr(server, "judge_json", lambda s, u, *a: {"score": 0.9, "reasoning": "fine"})
    body = json.loads(_call("evaluate_case", {"input": "q", "actual_output": "a", "metrics": ["toxicity"],
                                              "save": False}).content[0].text)
    assert body["verdict"]["toxicity"]["status"] == "pass"


def test_widget_inlines_the_shared_renderer():
    import server
    html = server._SCORES_HTML
    assert "window.RetrievalReport" in html or "root.RetrievalReport" in html
    assert ".rr-card" in html and "__RR_" not in html
    assert html.count("</script>") == 2  # nothing in report.js closes the tag early
    assert server.SCORES_WIDGET_URI.endswith("/v4")


def test_explain_rounds_like_the_page_and_names_p_contradicts():
    v = M.explain("faithfulness", [_row(0, "q", 0.625, "r")], 0.7)
    assert "averaged 0.63" in v["headline"]          # the page shows 0.63 too
    claims = [{"claim": "x", "p": 0.0, "bad": True, "verdict": "contradicts"}]
    v = M.explain("hallucination", [_row(0, "q", 0.0, "r", {"claims": claims, "kind": "contradiction"},
                                         "hallucination")], 0.7)
    assert "(P(contradicts)=1.00)" in v["drivers"][1]


def test_summary_card_single_case_with_claims():
    import server
    claims = [{"claim": "Bad one.", "p": 0.02, "bad": True, "label": "unsupported"},
              {"claim": "Iffy one.", "p": 0.59, "bad": False, "label": "supported"},
              {"claim": "Silent one.", "p": 1.0, "bad": False, "verdict": "says_nothing"},
              {"claim": "Good one.", "p": 0.97, "bad": False, "label": "supported"}]
    rows = [_row(0, "q", 0.67, "r", {"claims": claims, "kind": "support"})]
    agg = {"faithfulness": {"mean_score": 0.67, "verdict": M.explain("faithfulness", rows, 0.7)}}
    md = server._summary_md(agg, 0.7, 1, "", "https://x/report?run=1", rows)
    body = md.split("```diff\n", 1)[1].split("\n```", 1)[0].splitlines()
    assert body[0].startswith("  faithfulness 0.67 · FAIL")
    assert body[1:] == ["- ✕ 0.02  Bad one.",
                        "  ! 0.59  Iffy one. (weak support)",
                        "  ! 1.00  Silent one. (not in context)",
                        "+ ✓ 1 other claim supported by the context"]
