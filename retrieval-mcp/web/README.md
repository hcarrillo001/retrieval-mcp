# RetriEval — web (Vercel)

Two pages + one API, deployed as one Vercel project (Root Directory: `web`).

- `index.html`     → landing page (retrieval-mcp.com)
- `dashboard.html` → live dashboard (retrieval-mcp.com/dashboard), reads /api/runs
- `api/runs.js`    → serverless: reads run history from Supabase (service key server-side)
- `vercel.json`    → cleanUrls (so /dashboard serves dashboard.html)

## Deploy
Import the repo to Vercel, set Root Directory to `web`, framework "Other".
Env vars: SUPABASE_URL, SUPABASE_SERVICE_KEY, RETRIEVAL_RUNS_TABLE=runs
(optional DASH_TOKEN to gate /api/runs). See ../DEPLOY.md for the full walkthrough.
