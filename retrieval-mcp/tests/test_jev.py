"""Jev backend tests. No network and no API key: a local HTTP server stands in
for api.typesafe.ai, and the claim-splitting LLM is a stub.

    cd retrieval-mcp && python -m pytest tests/ -q
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUESTS = []


class FakeJev(BaseHTTPRequestHandler):
    """Supports anything with 'bridge' in the context, except claims about 1937."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"auth": self.headers.get("Authorization"), "body": body})
        if self.headers.get("Authorization") != "Bearer test-key":
            self.send_response(401); self.end_headers(); return
        answers = {}
        for name, q in body["questions"].items():
            if q["type"] == "noul":
                claim = body["state"]["claim"]
                answers[name] = {"type": "noul",
                                 "noul": 0.03 if "1937" in claim else 0.97}
            elif q["type"] == "choice":
                claim = body["state"]["claim"]
                pick = ("contradicts" if "2,000" in claim else
                        "says_nothing" if "1937" in claim else "supports")
                probs = {k: (0.9 if k == pick else 0.05) for k in q["criteria"]}
                answers[name] = {"type": "choice", "choice": pick,
                                 "probabilities": probs, "confidence": 0.8}
            elif q["type"] == "score":
                n = len(q["criteria"])
                answers[name] = {"type": "score", "score": 3.4, "confidence": 0.61,
                                 "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                 "probabilities": {str(i): (0.6 if i == 3 else 0.4 if i == 4 else 0.0)
                                                   for i in range(n)}}
        out = json.dumps({"model": "jev-test", "answers": answers,
                          "usage": {"input_tokens": 1_000_000, "output_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture()
def jev_env(tmp_path, monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeJev)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("RETRIEVAL_HOME", str(tmp_path))
    monkeypatch.setenv("RETRIEVAL_JUDGE_BACKEND", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_URL", f"http://127.0.0.1:{srv.server_port}/v1/systemone")
    monkeypatch.delenv("RETRIEVAL_BUDGET_USD", raising=False)
    REQUESTS.clear()
    yield
    srv.shutdown()


GG_CONTEXT = ["The Golden Gate Bridge has a main span of 1,280 metres."]


def stub_jj(claims):
    return lambda system, user, *a, **k: {"claims": claims}


def test_faithfulness_golden_gate_fixture(jev_env):
    import judge, metrics as M
    case = {"input": "Tell me about the Golden Gate Bridge.",
            "actual_output": "It has a 1,280 m main span. It opened in 1937.",
            "retrieval_context": GG_CONTEXT}
    r = M.faithfulness_jev(case, stub_jj(["The Golden Gate Bridge has a 1,280 m main span.",
                                          "The Golden Gate Bridge opened in 1937."]),
                           judge.jev_ask, judge.jev_map)
    assert r["score"] == 0.5  # matches the LLM-judge fixture: case 2 = 0.50
    assert r["details"]["unsupported_claims"] == ["The Golden Gate Bridge opened in 1937."]
    assert "p=0.03" in r["reason"] and "1 of 2" in r["reason"]
    # the API contract: object state, noul question, bearer auth, jev-latest
    b = REQUESTS[0]["body"]
    assert REQUESTS[0]["auth"] == "Bearer test-key"
    assert b["model"] == "jev-latest"
    assert set(b["state"]) == {"context", "claim"}
    assert b["questions"]["supported"]["type"] == "noul"


def test_faithfulness_no_claims_is_perfect_and_skips_jev(jev_env):
    import judge, metrics as M
    r = M.faithfulness_jev({"actual_output": "Hi!", "retrieval_context": GG_CONTEXT},
                           stub_jj([]), judge.jev_ask, judge.jev_map)
    assert r["score"] == 1.0 and REQUESTS == []


def test_answer_relevancy_score_normalised(jev_env):
    import judge, metrics as M
    r = M.answer_relevancy_jev({"input": "q", "actual_output": "a"}, None, judge.jev_ask)
    assert r["score"] == pytest.approx(3.4 / 4, abs=1e-4)
    assert "level 3/4" in r["reason"]
    assert len(REQUESTS[0]["body"]["questions"]["relevancy"]["criteria"]) == 5


def test_spend_is_metered_on_input_tokens(jev_env):
    import judge
    judge.reset_spend()
    judge.jev_ask({"context": "bridge", "claim": "x"},
                  {"s": {"type": "noul", "instructions": "i"}})
    assert judge.get_spend() == pytest.approx(0.042)


def test_budget_cap_blocks_jev(jev_env, monkeypatch):
    import judge
    judge.reset_spend()
    monkeypatch.setenv("RETRIEVAL_BUDGET_USD", "0.04")
    judge.jev_ask({"context": "bridge", "claim": "x"}, {"s": {"type": "noul", "instructions": "i"}})
    with pytest.raises(judge.BudgetExceeded):
        judge.jev_ask({"context": "bridge", "claim": "x"}, {"s": {"type": "noul", "instructions": "i"}})


def test_bad_key_gives_readable_error(jev_env, monkeypatch):
    import judge
    monkeypatch.setenv("TYPESAFE_API_KEY", "wrong")
    with pytest.raises(RuntimeError, match="rejected the API key"):
        judge.jev_ask("s", {"s": {"type": "noul", "instructions": "i"}})


def test_server_routes_by_backend(jev_env, monkeypatch):
    import server, metrics as M
    monkeypatch.setattr(server, "judge_json", stub_jj(["The Golden Gate Bridge opened in 1937."]))
    case = {"input": "q", "actual_output": "It opened in 1937.", "retrieval_context": GG_CONTEXT}
    r = server._run_metric("faithfulness", case, 0.7)
    assert r["details"]["judge"] == "jev" and r["score"] == 0.0
    assert server.list_metrics()["scored_by_jev"] == ["answer_relevancy", "faithfulness", "hallucination"]
    # a metric with no jev path goes to the LLM judge
    monkeypatch.setattr(server, "judge_json", lambda s, u, *a: {"score": 0.9, "reasoning": "ok"})
    assert server._run_metric("toxicity", case, 0.7)["score"] == 0.9
    # sandbox override always bypasses jev
    monkeypatch.setenv("GROQ_API_KEY", "g")
    import judge
    with judge.judge_as("groq-llama"):
        assert not judge.jev_enabled()


def test_default_backend_unchanged(monkeypatch):
    import judge
    monkeypatch.delenv("RETRIEVAL_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("TOUCHSTONE_JUDGE_BACKEND", raising=False)
    assert not judge.jev_enabled() and judge.get_judge() is judge._anthropic
    monkeypatch.setenv("RETRIEVAL_JUDGE_BACKEND", "jev")
    monkeypatch.setenv("RETRIEVAL_DECOMPOSER_BACKEND", "ollama")
    assert judge.get_judge() is judge._ollama


# ---- sandbox (the site's "Try it" box) -------------------------------------

def _stub_decomposer(monkeypatch, claims):
    """The sandbox splits claims with the Groq preset via judge._openai."""
    import judge
    seen = {}

    def fake_openai(system, user, max_tokens=1024):
        seen["model"] = judge._override.get()["model"]
        return json.dumps({"claims": claims})
    monkeypatch.setattr(judge, "_openai", fake_openai)
    return seen


def test_sandbox_lists_jev_only_when_key_set(monkeypatch):
    import judge
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert "jev" not in [m["id"] for m in judge.sandbox_models()]
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    jev = [m for m in judge.sandbox_models() if m["id"] == "jev"][0]
    assert jev["available"] and jev["metrics"] == ["answer_relevancy", "faithfulness", "hallucination"]


def test_sandbox_jev_faithfulness_end_to_end(jev_env, monkeypatch):
    import server
    monkeypatch.setenv("RETRIEVAL_JUDGE_BACKEND", "anthropic")  # server default is NOT jev
    monkeypatch.setenv("GROQ_API_KEY", "g")
    seen = _stub_decomposer(monkeypatch, ["The Golden Gate Bridge has a 1,280 m main span.",
                                          "The Golden Gate Bridge opened in 1937."])
    out = server.run_sandbox_eval(
        [{"input": "Tell me about the bridge.", "actual_output": "1,280 m span; opened 1937.",
          "retrieval_context": GG_CONTEXT}], "faithfulness", "jev")
    assert "error" not in out, out
    assert out["aggregate"]["faithfulness"]["mean_score"] == 0.5
    assert out["per_case"][0]["scores"]["faithfulness"]["details"]["judge"] == "jev"
    assert out["judge_model"].startswith("jev-latest + ")
    assert seen["model"] == server.SANDBOX_PRESETS["groq-llama"]["model"]
    assert len(REQUESTS) == 2  # one Jev call per claim


def test_sandbox_jev_rejects_metrics_it_cannot_score(jev_env, monkeypatch):
    import server
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setattr(server, "SANDBOX_ALLOWED_METRICS",
                        server.SANDBOX_ALLOWED_METRICS | {"toxicity"})
    out = server.run_sandbox_eval([{"input": "q", "actual_output": "a",
                                    "retrieval_context": ["c"]}], "toxicity", "jev")
    assert out["error"] == "metric_not_supported_by_judge" and REQUESTS == []


def test_sandbox_jev_hallucination(jev_env, monkeypatch):
    import server
    monkeypatch.setenv("GROQ_API_KEY", "g")
    _stub_decomposer(monkeypatch, [
        {"claim": "The Golden Gate Bridge has a 2,000 m main span.", "quote": "2,000 m span"},
        {"claim": "The Golden Gate Bridge opened in 1937.", "quote": "opened 1937"}])
    out = server.run_sandbox_eval([{"input": "q", "actual_output": "2,000 m span; opened 1937.",
                                    "retrieval_context": GG_CONTEXT}], "hallucination", "jev")
    assert "error" not in out, out
    d = out["per_case"][0]["scores"]["hallucination"]["details"]
    assert out["aggregate"]["hallucination"]["mean_score"] == 0.5  # one of two contradicts
    assert [r["label"] for r in d["claims"]] == ["contradicts", "not in context"]
    assert d["claims"][0]["quote"] == "2,000 m span" and d["not_in_context"] == 1


def test_sandbox_groq_preset_ignores_jev_backend(jev_env, monkeypatch):
    import server
    monkeypatch.setenv("GROQ_API_KEY", "g")  # RETRIEVAL_JUDGE_BACKEND=jev from fixture
    monkeypatch.setattr(server, "judge_json", lambda s, u, *a: {"score": 0.9, "reasoning": "ok"})
    out = server.run_sandbox_eval([{"input": "q", "actual_output": "a",
                                    "retrieval_context": ["c"]}], "faithfulness", "groq-llama")
    assert out["aggregate"]["faithfulness"]["mean_score"] == 0.9 and REQUESTS == []


# ---- Claude / any MCP client: the `judge` argument ---------------------------

def _call(tool, args):
    import asyncio
    import server
    from mcp.shared.memory import create_connected_server_and_client_session as conn

    async def go():
        async with conn(server.mcp._mcp_server) as c:
            return await c.call_tool(tool, args)
    return asyncio.run(go())


def test_mcp_run_eval_judge_jev_per_call(jev_env, monkeypatch):
    import server
    monkeypatch.setenv("RETRIEVAL_JUDGE_BACKEND", "anthropic")
    monkeypatch.setattr(server, "judge_json", stub_jj(["The Golden Gate Bridge opened in 1937."]))
    cases = json.dumps([{"input": "q", "actual_output": "It opened in 1937.",
                         "retrieval_context": GG_CONTEXT}])
    r = _call("run_eval", {"metrics": ["faithfulness"], "cases": cases, "judge": "jev"})
    sc = r.structuredContent
    assert not r.isError and sc["aggregate"]["faithfulness"]["mean_score"] == 0.0
    assert sc["judge_model"].startswith("jev-latest")
    assert sc["per_case"][0]["scores"]["faithfulness"]["details"]["claims"][0]["p_supported"] == 0.03


def test_mcp_judge_llm_overrides_jev_default(jev_env, monkeypatch):
    import server
    monkeypatch.setattr(server, "judge_json", lambda s, u, *a: {"score": 0.8, "reasoning": "ok"})
    r = _call("evaluate_case", {"input": "q", "actual_output": "a", "metrics": ["faithfulness"],
                                "retrieval_context": ["c"], "judge": "llm", "save": False})
    body = json.loads(r.content[0].text)
    assert body["scores"]["faithfulness"]["score"] == 0.8 and REQUESTS == []


def test_mcp_judge_env_default_is_jev(jev_env, monkeypatch):
    import server
    monkeypatch.setattr(server, "judge_json", stub_jj(["The bridge span is 1,280 m."]))
    r = _call("evaluate_case", {"input": "q", "actual_output": "a", "metrics": ["faithfulness"],
                                "retrieval_context": GG_CONTEXT, "save": False})
    body = json.loads(r.content[0].text)
    assert body["scores"]["faithfulness"]["details"]["judge"] == "jev"


def test_mcp_judge_jev_without_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("RETRIEVAL_HOME", str(tmp_path))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    r = _call("evaluate_case", {"input": "q", "actual_output": "a", "metrics": ["faithfulness"],
                                "retrieval_context": ["c"], "judge": "jev", "save": False})
    assert r.isError and "TYPESAFE_API_KEY" in r.content[0].text


# ---- quotes (for highlighting the answer) and hallucination -------------------

def test_quotes_are_located_in_the_answer():
    import metrics as M
    ans = "It has a 1,280 m  main span. It opened in 1937."
    jj = lambda s, u, *a: {"claims": [
        {"claim": "The bridge has a 1,280 m main span.", "quote": "1,280 m main span"},  # spacing drift
        {"claim": "The bridge opened in 1937.", "quote": "IT OPENED IN 1937"},          # case drift
        {"claim": "Invented.", "quote": "not in the answer"},
        "Plain string claim"]}
    got = M.extract_claims_with_quotes({"actual_output": ans}, jj)
    assert [g["quote"] for g in got] == ["1,280 m  main span", "It opened in 1937", None, None]
    assert all(g["quote"] is None or g["quote"] in ans for g in got)


def test_hallucination_jev_scores_only_contradictions(jev_env):
    import judge, metrics as M
    r = M.hallucination_jev(
        {"actual_output": "x", "retrieval_context": GG_CONTEXT},
        stub_jj(["The main span is 2,000 metres.", "It opened in 1937.", "It is a bridge."]),
        judge.jev_ask, judge.jev_map)
    assert r["score"] == pytest.approx(2 / 3, abs=1e-4)
    assert r["details"]["contradictions"] == ["The main span is 2,000 metres."]
    assert r["details"]["not_in_context"] == 1
    assert "1 of 3 claims contradict" in r["reason"]
    q = REQUESTS[0]["body"]["questions"]["relation"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"supports", "contradicts", "says_nothing"}


def test_faithfulness_rows_carry_display_fields(jev_env):
    import judge, metrics as M
    r = M.faithfulness_jev({"actual_output": "It opened in 1937.", "retrieval_context": GG_CONTEXT},
                           stub_jj([{"claim": "The bridge opened in 1937.", "quote": "It opened in 1937."}]),
                           judge.jev_ask, judge.jev_map)
    row = r["details"]["claims"][0]
    assert row["bad"] and row["label"] == "unsupported" and row["quote"] == "It opened in 1937."
    assert r["details"]["kind"] == "support"
