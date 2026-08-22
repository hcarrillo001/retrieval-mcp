# retriEVAL

**LLM evaluation as an MCP server.** Score your AI's outputs for faithfulness,
relevancy, and hallucination from inside any MCP client — no pipeline, no test
harness. Every result comes back with a link to a dashboard that keeps the history.

**[Try it live](https://retrieval-mcp.com)** (no signup) · **[Watch the 2-minute demo](https://youtu.be/zpXiv9isDmg)** · **[Dashboard](https://retrieval-mcp.com/dashboard)**

---

## Why

Five customer-support answers, scored on two metrics:

| metric | score | passing |
|---|---|---|
| answer_relevancy | 0.98 | 5/5 |
| faithfulness | 0.70 | 3/5 |

Every answer was on-topic and well-written. Two of them contradicted the policy
they were supposedly grounded in — one promised free return shipping the policy
doesn't offer, another invented a free overnight replacement. Reviewing by eye,
you'd sign off on all five.

That gap is the point. Relevancy asks *did it answer the question*. Faithfulness
asks *is it actually in the source*. You need both, and the second one catches
the expensive failures.

## Connect

Add as a custom connector in Claude, Claude Desktop, or any MCP client:

```
https://retrieval-mcp-production.up.railway.app/mcp?key=YOUR_TOKEN
```

The token goes in the URL rather than a header — chat clients don't offer
request-header auth. Then just ask:

> Score this answer against my docs with faithfulness.

Or run locally over stdio for development — see [Run locally (stdio)](#run-locally-stdio--claude-desktop) below.

## What you get

- **9 built-in metrics** plus custom metrics you author in plain English
- **Swappable judges** — Anthropic, Groq, Gemini, OpenRouter, or a local Ollama
  model, so nothing has to leave your network
- **Golden sets** from files, URLs, inline JSON, JSONL, CSV, or TSV
- **Run history** in Supabase with shareable permalinks and run comparison
- **A spend cap**, because a judge-based tool can otherwise run up a bill

## Honest limitations

- Judge agreement hasn't been validated against human labels yet, so treat
  scores as a signal rather than ground truth.
- Golden sets currently hold their own outputs, so re-running one against new
  model outputs means loading a second set. Splitting them is the next change.
- Golden sets and authored metrics live in the server process and are lost on
  restart. Runs persist; those don't.

---

## Metrics (DeepEval-aligned)

`faithfulness` · `answer_relevancy` · `contextual_precision` ·
`contextual_recall` · `contextual_relevancy` · `hallucination` · `bias` ·
`toxicity` · `summarization` — plus **authored G-Eval** metrics you define in
plain language. All are normalized so **higher = better** (bias/toxicity report
the clean fraction), and each reasons before scoring.

## Versatile golden sets

`load_golden_set` accepts a **file path** (including uploaded files), an
**http(s) URL**, an **inline JSON array**, or **JSONL text**, in
JSON / JSONL / CSV / TSV. Field names are auto-normalized (`question`→input,
`answer`→actual_output, `ground_truth`→expected_output, `contexts`→context,
`passages`→retrieval_context, …), so most public benchmarks load as-is.

## Short replies by default

`run_eval` scores every case but returns only the **3 lowest-scoring** by
default (tune with `limit`), with `total_cases`/`shown` and a pointer to
`show_run_cases(run_id, offset, limit, metric)` to page through the rest.

## Spend cap (so it can't run up a bill)

Judge spend is metered from real token usage and persisted. Set a hard cap:

```bash
export RETRIEVAL_BUDGET_USD=20     # 0/unset = unlimited
```

Once cumulative spend hits the cap, further Anthropic calls stop and tools
return a clear `budget_exceeded` message. Check/clear with `get_budget` /
`reset_budget`. (Prices are approximate — override `RETRIEVAL_PRICE_IN/OUT`
$/1M tokens to match current pricing for your model.)

## Hybrid setup (local + always-on dashboard)

Run history uses a pluggable store, chosen by env:
- **FileStore** (default) — JSONL in `~/.retrieval`. Zero setup, local only.
- **SupabaseStore** — when `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` are set. Run
  history lives in Postgres, shared by the local CLI, the deployed MCP, and the
  website dashboard.

Recommended hybrid flow:
1. Run `supabase_schema.sql` in Supabase (creates the `runs` table).
2. Set `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` on the MCP (local and/or Railway)
   so every run is written centrally. Each run records its `generator_model` and
   `judge_model` for cross-model comparison.
3. Deploy `web/` to Vercel (set the same Supabase env vars) and map it to
   `retrieval-mcp.com`. The dashboard reads history via `/api/runs` (service key
   stays server-side) and renders trend-by-model, a model leaderboard, and run
   history. It shows sample data until Supabase is wired.

Local stays your free sandbox (Ollama judge, file history); the website is the
always-on window into the shared history.

## Run locally (stdio) — Claude Desktop

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

```json
{
  "mcpServers": {
    "retrieval": {
      "command": "python",
      "args": ["/ABSOLUTE/PATH/server.py"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-...", "RETRIEVAL_BUDGET_USD": "10" }
    }
  }
}
```

Then: *"Load examples/rag_golden.jsonl as 'space', run faithfulness, label it v1."*

## Deploy as HTTP (reach it from anywhere)

```bash
export RETRIEVAL_TOKEN=$(openssl rand -hex 24)   # required for a public endpoint
export ANTHROPIC_API_KEY=sk-ant-...
export RETRIEVAL_BUDGET_USD=20
python app.py        # serves $PORT (default 8000); MCP at /mcp, health at /healthz
```

Deploy to **Railway** (or any host): the included `Dockerfile` / `Procfile`
work as-is. Set `ANTHROPIC_API_KEY`, `RETRIEVAL_TOKEN`, `RETRIEVAL_BUDGET_USD`
in the host env. Clients connect to `https://<host>/mcp` with header
`Authorization: Bearer <token>` — add it as a custom connector in
claude.ai / Claude Desktop, or point Agent Builder / CI at it. State (golden
sets, run history, spend) lives server-side, so it persists across machines.

## See it on a real RAG pipeline (demo)

`demo/rag_demo.py` builds a tiny end-to-end RAG over a small *labeled* dataset
(`demo/labeled.json` + `demo/corpus.json`): it retrieves with BM25, computes
`recall@k` against the gold passages (deterministic — the retriever's score),
generates an answer, then scores faithfulness (the generator's score). One answer
is deliberately hallucinated so you watch the two failure modes separate.

```bash
python demo/rag_demo.py            # offline, no key needed
python demo/rag_demo.py --real     # real generation + RetriEval judge (needs ANTHROPIC_API_KEY)
```

It also writes `demo/generated_goldenset.jsonl` — load *that* into the MCP
(`load_golden_set` → `run_eval`) for the judge-scored version. This is the bridge:
your pipeline emits predictions, the dataset supplies the labels, and RetriEval
scores retriever and generator independently.

## Connecting to a RAG pipeline

- **Offline (default):** export your pipeline's retrieved context + answer into a
  golden set and score it — RetriEval never touches your pipeline.
- **Live:** add a `query_rag(question)` tool that calls your RAG endpoint or your
  vector store (Chroma / Supabase pgvector), captures context + answer, and scores
  in one shot.

## Judge backend

```bash
export RETRIEVAL_JUDGE_BACKEND=anthropic          # default
export RETRIEVAL_JUDGE_MODEL=claude-sonnet-4-6
# or local, free:
export RETRIEVAL_JUDGE_BACKEND=ollama
export RETRIEVAL_JUDGE_MODEL=deepseek-r1:70b
```

## Tools

| Tool | Purpose |
|------|---------|
| `list_metrics` | built-in + authored metrics |
| `load_golden_set(name, source, fmt)` | file / URL / inline / JSONL-CSV |
| `list_golden_sets` | what's loaded |
| `author_metric(name, criteria, examples)` | plain language → a scorer |
| `run_eval(golden_set, metrics, threshold, outputs, label, limit)` | score; shows top 3 |
| `show_run_cases(run_id, offset, limit, metric)` | page the rest |
| `evaluate_case(...)` | one-off score |
| `ground_against_url(url, output, question)` | check an output's *consistency* with a web page (no labels — consistency, not correctness) |
| `list_runs(golden_set, last_n)` | saved runs |
| `plot_metric_trend / plot_run / compare_runs` | inline charts |
| `get_budget / reset_budget` | spend cap status / reset |

---

## License

Apache License 2.0. Built by Hanns Carrillo.
