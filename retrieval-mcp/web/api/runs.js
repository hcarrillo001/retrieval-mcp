// Vercel serverless function — GET /api/runs
// Reads run history from Supabase using the service key (kept server-side).
// Optional: set DASH_TOKEN and call /api/runs?key=... to gate access.
export default async function handler(req, res) {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  const table = process.env.RETRIEVAL_RUNS_TABLE || "runs";
  const gate = process.env.DASH_TOKEN;

  if (gate && req.query.key !== gate) {
    return res.status(401).json({ error: "unauthorized" });
  }
  if (!url || !key) {
    return res.status(500).json({ error: "SUPABASE_URL / SUPABASE_SERVICE_KEY not set" });
  }

  const gs = req.query.golden_set ? `&golden_set=eq.${encodeURIComponent(req.query.golden_set)}` : "";
  const limit = Math.min(parseInt(req.query.limit || "200", 10), 1000);

  try {
    const r = await fetch(
      `${url}/rest/v1/${table}?select=*&order=timestamp.desc&limit=${limit}${gs}`,
      { headers: { apikey: key, Authorization: `Bearer ${key}` } }
    );
    if (!r.ok) {
      return res.status(502).json({ error: "supabase error", status: r.status });
    }
    const data = await r.json();
    res.setHeader("Cache-Control", "s-maxage=30, stale-while-revalidate=60");
    return res.status(200).json(data);
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
