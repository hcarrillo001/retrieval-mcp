// Vercel serverless function — POST /api/chat
// Powers the dashboard's "Ask RetriEval" chat. Calls the Anthropic Messages API
// with your RetriEval MCP attached as a remote connector, so Claude can run evals
// and return charts. Your ANTHROPIC_API_KEY stays server-side.
//
// Env (set in Vercel):
//   ANTHROPIC_API_KEY    required — enables the chat (omit to disable it)
//   RETRIEVAL_MCP_URL    your deployed MCP endpoint, e.g. https://app.up.railway.app/mcp
//   RETRIEVAL_TOKEN      bearer token for the MCP (if your server requires it)
//   DASH_TOKEN           optional gate — if set, callers must send Authorization: Bearer <DASH_TOKEN>
//                        (the dashboard reads it from /dashboard?key=...). Strongly recommended
//                        so visitors can't spend your API key.
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const key = process.env.ANTHROPIC_API_KEY;
  const mcpUrl = process.env.RETRIEVAL_MCP_URL;
  const mcpToken = process.env.RETRIEVAL_TOKEN;
  const gate = process.env.DASH_TOKEN;

  if (gate) {
    const auth = req.headers["authorization"] || "";
    if (auth !== `Bearer ${gate}`) return res.status(401).json({ error: "unauthorized — open /dashboard?key=YOUR_DASH_TOKEN" });
  }
  if (!key) return res.status(503).json({ error: "chat disabled: ANTHROPIC_API_KEY not set" });

  let body = req.body;
  if (typeof body === "string") { try { body = JSON.parse(body || "{}"); } catch { body = {}; } }
  const messages = Array.isArray(body.messages) ? body.messages : [];

  const payload = {
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: "You are RetriEval's assistant. Use the retrieval MCP tools to run evals, author metrics, and plot charts. Be concise: show scores with pass/fail, and render charts when asked for trends or comparisons.",
    messages,
  };
  if (mcpUrl) {
    payload.mcp_servers = [{
      type: "url", url: mcpUrl, name: "retrieval",
      ...(mcpToken ? { authorization_token: mcpToken } : {}),
    }];
  }

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "mcp-client-2025-04-04",
      },
      body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) return res.status(502).json({ error: "anthropic error", detail: data });
    return res.status(200).json({ content: data.content });
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
