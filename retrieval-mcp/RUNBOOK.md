# RetriEval — end-to-end demo runbook

The whole flow, one document. Cursor orchestrates: it drives **Open WebUI** (the
system under test, running Claude-or-DeepSeek via Ollama) with Playwright,
captures the answer, scores it with your **RetriEval** MCP over HTTP, writes the
result to **Jira**, and every run lands in **Supabase** so the **dashboard** at
retrieval-mcp.com always shows the charts and history. A second, simpler demo
connects RetriEval to **Claude.ai** so manual testers can run evals by chatting.

```
                          ┌─────────────────────────── CURSOR (orchestrator) ───────────────────────────┐
                          │  Claude agent + the ticket-test-runner skill                                  │
                          │                                                                               │
   you: "test PROJ-123" ─▶│  1) Jira MCP      ── read ticket (url, steps, criteria)                       │
                          │  2) Playwright MCP ── drive Open WebUI, capture answer (+citations)           │
                          │  3) RetriEval MCP  ── score answer  ──────────────┐                           │
                          │  4) Jira MCP       ── comment + transition + chart│                           │
                          └───────────────────────────────────────────────────┼───────────────────────────┘
                                      │                  │                     │
                            ┌─────────▼────────┐ ┌───────▼────────┐   ┌────────▼─────────┐
                            │  Open WebUI       │ │  RetriEval     │   │  Jira / Xray      │
                            │  (SUT) + Ollama   │ │  HTTP @Railway │   │                   │
                            │  Claude/DeepSeek  │ │  + bearer token│   └───────────────────┘
                            └───────────────────┘ └───────┬────────┘
                                                          │ writes run history
                                                  ┌───────▼────────┐      ┌───────────────────┐
                                                  │ Supabase (runs) │─────▶│ Dashboard (Vercel) │
                                                  └─────────────────┘      │ retrieval-mcp.com  │
                                                                           │ charts + history   │
                                                                           └───────────────────┘
```

---

## Components & roles

| Component | Role | Hosting |
|-----------|------|---------|
| Cursor | Orchestrator (MCP host, runs the skill) | your machine |
| Playwright MCP | Drives a browser against the SUT | local (npx) |
| Open WebUI + Ollama | System under test: a real chat w/ RAG | local Docker or a cloud box |
| RetriEval MCP | Scores outputs, charts, history | **HTTP on Railway** + bearer token |
| Supabase | Shared run history (Postgres) | Supabase cloud |
| Dashboard | Charts + history view | Vercel → retrieval-mcp.com |
| Jira (+ Xray) | Tickets in/out | Atlassian cloud |

---

## Part A — Deploy RetriEval over HTTP (once)

1. `pip install -r requirements.txt`
2. In Supabase: run `supabase_schema.sql` (creates the `runs` table).
3. Deploy the repo to Railway (Dockerfile included). Set env:
   - `ANTHROPIC_API_KEY` — only if you judge with Claude. For **$0**, skip it and
     use Ollama: `RETRIEVAL_JUDGE_BACKEND=ollama`, `RETRIEVAL_JUDGE_MODEL=deepseek-r1:70b`
     (needs Ollama reachable from the server; for fully-local judging run the MCP
     locally instead — see note).
   - `RETRIEVAL_TOKEN` — `openssl rand -hex 24`
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` — so runs persist centrally
4. Health check: `GET https://<app>.up.railway.app/healthz` → `ok`.
   MCP endpoint: `https://<app>.up.railway.app/mcp` (send `Authorization: Bearer <token>`).

> Judge note: a cloud-hosted Ollama is extra infra. The simplest $0 judge path is
> to run RetriEval **locally over stdio** with Ollama on your Mac for Demo 1, and
> use the HTTP deploy (Claude judge or a hosted model) for Demo 2 / the dashboard.
> Both write to the same Supabase, so history stays unified either way.

## Part B — Open WebUI as the system under test (once)

```bash
docker run -d -p 3000:8080 --name openwebui \
  -v openwebui:/app/backend/data ghcr.io/open-webui/open-webui:main
```
- Point it at a model: Ollama (`ollama run deepseek-r1`) or an OpenAI-compatible
  endpoint for Claude. Open WebUI accepts any OpenAI-compatible base URL + key.
- (Optional RAG) Upload a doc in Open WebUI so answers are retrieval-grounded —
  that's what makes faithfulness/context metrics meaningful.
- It's now at `http://localhost:3000`. This is the URL your ticket points at.

## Part C — Cursor MCP config

`~/.cursor/mcp.json` (verify exact package names/flags against each project's
current README — they drift):

```json
{
  "mcpServers": {
    "playwright": { "command": "npx", "args": ["@playwright/mcp@latest"] },
    "retrieval": {
      "command": "npx",
      "args": ["mcp-remote", "https://<app>.up.railway.app/mcp",
               "--header", "Authorization: Bearer <YOUR_TOKEN>"]
    },
    "atlassian": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.atlassian.com/v1/mcp/authv2"]
    }
  }
}
```
Then drop the `ticket-test-runner` skill into Cursor. Playwright is local;
RetriEval and Jira are remote (brokered through `mcp-remote`).

---

## Part D — Demo 1: automated loop (the money shot)

In Cursor: **"test PROJ-123."** The skill runs:
1. Jira → read ticket (url = your Open WebUI, steps, expected, eval criteria).
2. Playwright → open Open WebUI, type the prompt, wait for the answer to finish,
   capture the answer text (and the citations Open WebUI shows, for context metrics).
3. RetriEval → `evaluate_case` (faithfulness, answer_relevancy, …) at the
   ticket's threshold; pass `generator_model` (the SUT model) + `judge_model`.
4. Jira → post the verdict comment **with the dashboard run link** and transition
   status — after you confirm.

Every run is saved to Supabase → visible on the dashboard instantly.

## Part E — Demo 2: manual evals inside Claude.ai

For non-coders. In Claude.ai → Settings → Connectors → Add custom connector →
paste `https://<app>.up.railway.app/mcp` (+ token). Now a tester just types:
- "Author a 'tone' metric that checks the answer is polite and concise."
- "Score this answer against this golden set: …"
- "Show me the faithfulness trend."
No Cursor, no Playwright. Same MCP, same Supabase history, same dashboard.
(Requires the **HTTP** deploy — claude.ai can't reach a local stdio server.)

---

## Part F — Evals, charts, history, and charts-in-Jira

- **Create evals**: `author_metric(name, criteria)` turns plain language into a
  scorer (works in both demos).
- **Charts**: `plot_metric_trend`, `plot_run`, `compare_runs` return PNGs inline
  in Cursor/Claude. The dashboard renders the same data live from Supabase.
- **History + dashboard**: every `run_eval` writes to Supabase; the dashboard
  shows KPIs, trend-by-model, the model leaderboard, and run history.
- **Charts into Jira** — two ways, easiest first:
  1. **Link** (always works): the result comment includes
     `https://retrieval-mcp.com/?run={run_id}` — one click to the run's charts.
     (Small dashboard enhancement: read `?run=` and scroll to that run.)
  2. **Attach the PNG**: `plot_run` returns a PNG; the orchestrator uploads it to
     the issue via Jira's attachments REST endpoint
     (`POST /rest/api/3/issue/{key}/attachments`, header
     `X-Atlassian-Token: no-check`). Do this only if your Jira MCP exposes an
     attachment tool; otherwise it's a small direct API call in the skill.
- **Xray**: to record a true pass/fail test execution (not just a comment), Xray
  has its own REST/GraphQL import API. Confirm whether a first-party Xray MCP
  exists; if not, the skill makes one Xray API call to create the execution.

---

## Run order (checklist)

1. [ ] Supabase: run `supabase_schema.sql`.
2. [ ] Deploy RetriEval to Railway (token + Supabase env). `/healthz` = ok.
3. [ ] Deploy `web/` to Vercel (same Supabase env) → map retrieval-mcp.com.
4. [ ] Start Open WebUI + a model; upload a doc for RAG.
5. [ ] Cursor `mcp.json`: playwright + retrieval + atlassian. Install the skill.
6. [ ] Create a sandbox Jira ticket with a `test` block pointing at Open WebUI.
7. [ ] In Cursor: "test <KEY>" → watch the loop → confirm the Jira write.
8. [ ] Open retrieval-mcp.com → the run + charts are there.
9. [ ] (Demo 2) Add RetriEval as a Claude.ai connector → run an eval by chatting.

## Honest constraints

- **Context metrics need the retrieved context.** Answer-only capture supports
  answer_relevancy; faithfulness / context_recall need Open WebUI's citations or
  an API call to fetch what it retrieved. Start answer-only, add context next.
- **Sandbox only.** Throwaway Jira + your own Open WebUI — never Litera/production
  (non-compete + clean public demo).
- **Package names drift.** Re-check the Playwright MCP, `mcp-remote`, and Atlassian
  MCP setup against their current READMEs before relying on the config above.
- **Local vs HTTP judge.** Fully-local $0 judging (Ollama) is easiest with the
  stdio server; the HTTP deploy is what powers Claude.ai (Demo 2) and the always-on
  dashboard. Both share Supabase, so it's one history.
