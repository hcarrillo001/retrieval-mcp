"""
RAG demo for RetriEval — see retriever failure and generator failure light up
on *different* metrics, using real labeled data.

What it does, per question:
  1. retrieve top-k passages from the corpus with BM25  (the "RAG db")
  2. recall@k  = of the gold passages, how many were retrieved   (DETERMINISTIC,
     no judge — this is the RETRIEVER's score)
  3. generate an answer from the retrieved context           (the GENERATOR)
  4. faithfulness = is the answer supported by what was retrieved?  (the
     GENERATOR's score — independent of recall)
  5. write a golden-set JSONL you can load into the MCP and score there too.

Run offline (no key needed) — generation + answer metrics use transparent local
approximations, and one answer is deliberately hallucinated to show the split:

    python demo/rag_demo.py

Run for real — uses Anthropic to generate answers AND RetriEval's judge metrics:

    export ANTHROPIC_API_KEY=sk-ant-...
    python demo/rag_demo.py --real
"""
from __future__ import annotations
import os
import re
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from retriever import BM25, tokenize  # noqa: E402

TOP_K = 2
REAL = "--real" in sys.argv and os.environ.get("ANTHROPIC_API_KEY")


# ---- generator -------------------------------------------------------------
def generate_real(question: str, context: list) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    model = os.environ.get("RETRIEVAL_JUDGE_MODEL", "claude-sonnet-4-6")
    ctx = "\n".join(f"- {c}" for c in context)
    msg = (f"Answer the question using ONLY the context. Be concise.\n\n"
           f"Context:\n{ctx}\n\nQuestion: {question}")
    r = client.messages.create(model=model, max_tokens=200,
                               messages=[{"role": "user", "content": msg}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


def generate_stub(qid: str, question: str, context: list) -> str:
    # deterministic, offline. q2 is deliberately hallucinated to show that a good
    # retrieval can still produce a bad (unfaithful) answer.
    if qid == "q2":
        return "Olympus Mons stands about 96 kilometers tall."  # WRONG (ctx says 22)
    return context[0] if context else "I don't know."


# ---- offline metric approximations (used when not --real) ------------------
def _nums(s):
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def faith_approx(answer: str, context: list) -> float:
    ctx = " ".join(context)
    a = _nums(answer)
    if not a:
        return 1.0
    return round(len(a & _nums(ctx)) / len(a), 2)


def relevancy_approx(answer: str, question: str) -> float:
    q = {w for w in tokenize(question) if len(w) > 3}
    a = set(tokenize(answer))
    return round(min(1.0, 0.5 + len(q & a) / max(len(q), 1) / 2), 2)


def recall_approx(expected: str, context: list) -> float:
    e = {w for w in tokenize(expected) if len(w) > 3}
    ctx = set(tokenize(" ".join(context)))
    if not e:
        return 1.0
    return round(len(e & ctx) / len(e), 2)


def main():
    corpus = json.loads((HERE / "corpus.json").read_text())
    data = json.loads((HERE / "labeled.json").read_text())
    bm25 = BM25().index(corpus)

    if REAL:
        from judge import judge_json
        import metrics as M

    rows, goldenset = [], []
    for item in data:
        q, gold_ans, gold_ps = item["question"], item["gold_answer"], item["gold_passages"]
        retrieved = [pid for pid, _ in bm25.query(q, k=TOP_K)]
        context = [bm25.texts[pid] for pid in retrieved]

        # 2) retriever score: recall@k over gold passages (deterministic)
        hit = sum(1 for g in gold_ps if g in retrieved)
        recall_at_k = round(hit / len(gold_ps), 2)

        # 3) generate
        answer = (generate_real(q, context) if REAL
                  else generate_stub(item["id"], q, context))

        # 4) generator scores
        case = {"input": q, "actual_output": answer,
                "retrieval_context": context, "expected_output": gold_ans}
        if REAL:
            faith = M.faithfulness(case, judge_json)["score"]
            relev = M.answer_relevancy(case, judge_json)["score"]
            crecall = M.contextual_recall(case, judge_json)["score"]
        else:
            faith = faith_approx(answer, context)
            relev = relevancy_approx(answer, q)
            crecall = recall_approx(gold_ans, context)

        rows.append((item["id"], retrieved, gold_ps, recall_at_k, faith, relev, crecall, answer))
        goldenset.append(case)

    # write a golden set the MCP can load + score
    out = HERE / "generated_goldenset.jsonl"
    out.write_text("\n".join(json.dumps(c) for c in goldenset))

    # ---- print ----
    mode = "REAL (Anthropic generate + RetriEval judge)" if REAL else "OFFLINE (stub + local approx)"
    print(f"\nRAG demo — mode: {mode}   top_k={TOP_K}\n" + "=" * 78)
    print(f"{'q':>3}  {'retrieved':<12} {'gold':<6} {'recall@k':>8} {'faith':>6} {'relev':>6} {'c_rec':>6}")
    print("-" * 78)
    for qid, retr, gold, rk, fa, re, cr, ans in rows:
        flag = ""
        if rk < 1.0:
            flag = "  <- RETRIEVER missed gold passage"
        elif fa < 0.7:
            flag = "  <- GENERATOR unfaithful (hallucinated)"
        print(f"{qid:>3}  {','.join(retr):<12} {','.join(gold):<6} {rk:>8} {fa:>6} {re:>6} {cr:>6}{flag}")
    print("-" * 78)
    print("Read it this way: recall@k is the RETRIEVER. faith is the GENERATOR.")
    print("They move independently — that's the point of evaluating RAG by stage.\n")
    print(f"Golden set written -> {out}")
    print("Next: in Claude Desktop, load that file and run faithfulness/contextual_recall\n"
          "through the MCP for the judge-scored version.")


if __name__ == "__main__":
    main()
