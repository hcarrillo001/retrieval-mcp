// Vercel serverless function — POST /api/sandbox
// Public eval sandbox. Firecrawl-style: real input allowed, but capped PER IP
// PER DAY (a request count) with a 429 when exceeded. Free-model keys live on
// the Railway server; this function only forwards a short model id + the cases.
//
// Env (Vercel):
//   SUPABASE_URL, SUPABASE_SERVICE_KEY     rate-limit store (table: sandbox_usage)
//   RETRIEVAL_SANDBOX_URL                  e.g. https://<host>.up.railway.app/sandbox
//   SANDBOX_SECRET                         shared secret with the Railway /sandbox route
//   SANDBOX_DAILY_LIMIT                    runs per IP per day (default 15)
//
// Supabase table:
//   create table if not exists sandbox_usage (
//     ip text, day date, count int default 0, primary key (ip, day));

const SAMPLES = {
  halluc: {
    metric: "faithfulness",
    cases: [{
      input: "What is the Eiffel Tower made of?",
      retrieval_context: ["The Eiffel Tower is built from puddled wrought iron."],
      actual_output: "The Eiffel Tower is made entirely of solid gold.",
    }],
  },
  clean: {
    metric: "faithfulness",
    cases: [{
      input: "What is the capital of France?",
      retrieval_context: ["Paris is the capital and most populous city of France."],
      actual_output: "The capital of France is Paris.",
    }],
  },
  mixed: {
    metric: "faithfulness",
    cases: [
      { input: "When was the Eiffel Tower completed?",
        retrieval_context: ["The Eiffel Tower was completed in 1889 for the World's Fair."],
        actual_output: "It was completed in 1889." },
      { input: "What is the Eiffel Tower made of?",
        retrieval_context: ["The Eiffel Tower is built from wrought iron."],
        actual_output: "It is made entirely of solid gold." },
      { input: "How many moons does Mars have?",
        retrieval_context: ["Mars has two small moons, Phobos and Deimos."],
        actual_output: "Mars has 12 moons, the largest named Titan." },
    ],
  },
};
const ALLOWED_METRICS = ["faithfulness", "answer_relevancy", "hallucination", "contextual_relevancy"];
const MAX_CHARS = 8000;   // keep in sync with SANDBOX_MAX_CHARS on the server

function clientIp(req) {
  const xf = req.headers["x-forwarded-for"];
  if (xf) return String(xf).split(",")[0].trim();
  return req.socket?.remoteAddress || "unknown";
}

async function checkAndBumpQuota(ip) {
  const url = process.env.SUPABASE_URL, key = process.env.SUPABASE_SERVICE_KEY;
  const limit = parseInt(process.env.SANDBOX_DAILY_LIMIT || "15", 10);
  if (!url || !key) return { ok: true, remaining: limit }; // no store → don't block
  const day = new Date().toISOString().slice(0, 10);
  const h = { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json" };
  try {
    const g = await fetch(
      `${url}/rest/v1/sandbox_usage?select=count&ip=eq.${encodeURIComponent(ip)}&day=eq.${day}`,
      { headers: h });
    const rows = g.ok ? await g.json() : [];
    const current = rows[0]?.count || 0;
    if (current >= limit) return { ok: false, remaining: 0 };
    // upsert count+1 (last-write-wins; fine for a demo)
    await fetch(`${url}/rest/v1/sandbox_usage?on_conflict=ip,day`, {
      method: "POST",
      headers: { ...h, Prefer: "resolution=merge-duplicates" },
      body: JSON.stringify([{ ip, day, count: current + 1 }]),
    });
    return { ok: true, remaining: limit - current - 1 };
  } catch (e) {
    return { ok: true, remaining: limit }; // never hard-fail the demo on store errors
  }
}

function clip(v) {
  if (Array.isArray(v)) return v.slice(0, 8).map((x) => String(x).slice(0, MAX_CHARS));
  return String(v == null ? "" : v).slice(0, MAX_CHARS);
}

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const serverUrl = process.env.RETRIEVAL_SANDBOX_URL;
  const secret = process.env.SANDBOX_SECRET;
  if (!serverUrl || !secret) {
    return res.status(503).json({ error: "sandbox not configured" });
  }

  // per-IP daily cap (Firecrawl-style)
  const ip = clientIp(req);
  const quota = await checkAndBumpQuota(ip);
  if (!quota.ok) {
    return res.status(429).json({ error: "daily_limit_reached",
      message: "You've hit today's free sandbox limit. Try again tomorrow." });
  }

  const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
  const model = body.model || "groq-llama";

  // resolve cases: a curated sample id, or custom user-supplied cases
  let cases, metric;
  if (body.sample && SAMPLES[body.sample]) {
    cases = SAMPLES[body.sample].cases;
    metric = body.metric || SAMPLES[body.sample].metric;
  } else if (Array.isArray(body.cases) && body.cases.length) {
    cases = body.cases.slice(0, 3).map((c) => ({
      input: clip(c.input),
      actual_output: clip(c.actual_output),
      expected_output: clip(c.expected_output),
      retrieval_context: clip(c.retrieval_context || c.context || []),
    }));
    metric = body.metric || "faithfulness";
  } else {
    return res.status(400).json({ error: "no_input", message: "Pick a sample or provide cases." });
  }
  if (!ALLOWED_METRICS.includes(metric)) {
    return res.status(400).json({ error: "metric_not_allowed", allowed: ALLOWED_METRICS });
  }

  try {
    const r = await fetch(serverUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-sandbox-secret": secret },
      body: JSON.stringify({ cases, metric, model }),
    });
    const data = await r.json();
    if (!r.ok) return res.status(r.status).json(data);
    return res.status(200).json({ ...data, remaining_today: quota.remaining });
  } catch (e) {
    return res.status(502).json({ error: "sandbox_server_error", detail: String(e) });
  }
}
