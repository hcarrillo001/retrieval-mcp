# Deploying RetriEval (Supabase + Railway + Vercel)

Three services. Do them in this order — Supabase first (both others need its
keys), then Railway (the MCP server), then Vercel (the dashboard).

Time: ~20–30 min. Cost: Supabase + Vercel have free tiers; Railway is a few $/mo;
LLM judging is $0 if you use Ollama, otherwise Anthropic API usage.

---

## 1. Supabase (shared run history) — do this first

1. Create a project at supabase.com (free tier is fine). Pick a region near you.
2. Open **SQL Editor** → paste the contents of `supabase_schema.sql` → **Run**.
   (Creates the `runs` table + indexes.)
3. Go to **Project Settings → API** and copy two values:
   - **Project URL** → this is `SUPABASE_URL` (e.g. `https://abcd.supabase.co`)
   - **service_role key** (under "Project API keys") → this is `SUPABASE_SERVICE_KEY`
   ⚠️ The service_role key is secret — server-side only, never in the browser.

---

## 2. Railway (the RetriEval MCP server)

1. Push this project to a GitHub repo (the folder with `app.py`, `Dockerfile`, etc.).
2. At railway.app → **New Project → Deploy from GitHub repo** → pick the repo.
   Railway detects the `Dockerfile` automatically.
3. **Variables** tab → add:
   ```
   RETRIEVAL_TOKEN          = (run: openssl rand -hex 24)
   SUPABASE_URL             = https://abcd.supabase.co
   SUPABASE_SERVICE_KEY     = (the service_role key)
   RETRIEVAL_RUNS_TABLE     = runs
   ```
   Then pick your judge:
   - Anthropic: `ANTHROPIC_API_KEY=sk-ant-...`, `RETRIEVAL_JUDGE_MODEL=claude-sonnet-4-6`
   - or free/local Ollama: `RETRIEVAL_JUDGE_BACKEND=ollama` (+ a reachable `OLLAMA_URL`)
   - optional cap: `RETRIEVAL_BUDGET_USD=20`
4. Railway sets `PORT` for you; `app.py` reads it. Deploy.
5. Under **Settings → Networking**, generate a public domain. Test:
   - `https://YOUR-APP.up.railway.app/healthz` → `ok`
   - Your MCP endpoint is `https://YOUR-APP.up.railway.app/mcp`
6. Save the URL + token — you'll use them in Cursor and in Claude.ai.

---

## 3. Vercel (the dashboard at retrieval-mcp.com)

1. At vercel.com → **Add New → Project** → import the same GitHub repo.
2. **Root Directory**: set it to `web` (the dashboard lives there).
   Framework preset: **Other** (it's static + a serverless function; no build step).
3. **Environment Variables** → add the same Supabase pair:
   ```
   SUPABASE_URL          = https://abcd.supabase.co
   SUPABASE_SERVICE_KEY  = (service_role key)
   RETRIEVAL_RUNS_TABLE  = runs
   ```
   (Optional: `DASH_TOKEN=secret` to gate `/api/runs`, then open `/?key=secret`.)
4. Deploy. You get a `*.vercel.app` URL — open it; it shows sample data until a
   real run lands, then live data from Supabase.
5. **Custom domain**: Project → **Settings → Domains** → add `retrieval-mcp.com`.
   Vercel shows the DNS records to set at your registrar (an `A`/`ALIAS` for the
   apex + a `CNAME` for `www`). Add them where you bought the domain; HTTPS is
   automatic once DNS propagates.

---

## 4. Verify the whole loop

1. Locally or from Cursor, run an eval through the MCP (e.g. load the demo golden
   set, `run_eval`). Because Supabase env is set, the run writes to Postgres.
2. Refresh the dashboard — the run + charts appear (header flips to "live · Supabase").
3. The Jira comment link `https://retrieval-mcp.com/?run={run_id}` highlights that
   run in history.

---

## Connecting clients (after deploy)

**Cursor** (`~/.cursor/mcp.json`) — see `RUNBOOK.md` Part C. RetriEval entry:
```json
"retrieval": { "command": "npx", "args": ["mcp-remote",
  "https://YOUR-APP.up.railway.app/mcp", "--header", "Authorization: Bearer YOUR_TOKEN"] }
```

**Claude.ai** (manual evals) — Settings → Connectors → Add custom connector →
URL `https://YOUR-APP.up.railway.app/mcp`. (Token via the connector's auth.)

---

## Gotchas

- **Supabase env on BOTH Railway and Vercel** — that shared table is what unifies
  history across the MCP and the dashboard. Miss it on one side and they diverge.
- **service_role key is secret.** It's safe in Railway/Vercel server env and in the
  `/api/runs` function; never expose it to the browser.
- **Railway public networking** must be enabled or Claude.ai/Cursor can't reach the
  MCP (it connects from Anthropic's cloud, not your laptop).
- **Free judging:** leave `ANTHROPIC_API_KEY` off and use Ollama to keep cost at $0;
  the API key (with its own spend limit) is the simplest hard ceiling otherwise.

---

## Optional: the dashboard chat ("Ask RetriEval")

The dashboard has an optional chat panel that lets you run evals and see charts
without leaving the page. It calls Claude server-side with your RetriEval MCP
attached, so your API key never reaches the browser. To enable it, add to the
**Vercel** project (in addition to the Supabase vars):

```
ANTHROPIC_API_KEY   = sk-ant-...            # enables the chat (omit to hide/disable it)
RETRIEVAL_MCP_URL   = https://YOUR-APP.up.railway.app/mcp
RETRIEVAL_TOKEN     = (same token as Railway)
DASH_TOKEN          = (a secret)            # STRONGLY recommended
```

With `DASH_TOKEN` set, open the chat at `https://retrieval-mcp.com/dashboard?key=YOUR_DASH_TOKEN`.
This keeps strangers from spending your API key. Without `ANTHROPIC_API_KEY`, the
dashboard still works fully as a metrics view; the chat just stays disabled.

### Three ways to use RetriEval
1. **All-in-one** — your deployed dashboard: charts/history *and* the chat panel.
2. **Inside Claude/Cursor** — add RetriEval as a connector; the chat is Claude itself.
3. **Metrics-only, no dashboard** — run the MCP locally (stdio) in Claude Desktop;
   you get scores and charts in the chat, history in a local file, no website needed.
