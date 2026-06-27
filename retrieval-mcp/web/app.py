"""
HTTP entrypoint for RetriEval — deploy this to reach the server from anywhere.

Wraps the MCP's streamable-HTTP ASGI app with bearer-token auth so a public
endpoint isn't open to the world (important, since each call spends API budget).

Local stdio mode is still `python server.py`. For HTTP:

    export RETRIEVAL_TOKEN=$(openssl rand -hex 24)
    export ANTHROPIC_API_KEY=sk-ant-...
    export RETRIEVAL_BUDGET_USD=20            # hard spend cap
    python app.py                             # serves on $PORT (default 8000)

Clients connect to  https://<host>/mcp  with header:  Authorization: Bearer <token>
A health check is exposed (unauthenticated) at  /healthz.

Note on hosting: FastMCP's streamable-HTTP transport enables DNS-rebinding
protection by default, which only trusts localhost. Behind a public host
(Railway/Render/Fly/etc.) the incoming Host header is your public domain, which
would otherwise be rejected with "421 Misdirected Request". We relax that here
so the server works behind a proxy. Set RETRIEVAL_ALLOWED_HOSTS (comma-separated)
to lock it down to specific hostnames if you prefer.
"""
from __future__ import annotations
import os

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from server import mcp

TOKEN = os.environ.get("RETRIEVAL_TOKEN", "")
PUBLIC_PATHS = ("/healthz",)

# --- Fix 421 Misdirected Request behind a public host -----------------------
# Allow the public host(s) the platform serves us on. Default: allow all hosts
# (proxy already terminates TLS and we enforce our own bearer auth below).
# Lock down by setting RETRIEVAL_ALLOWED_HOSTS="myhost.up.railway.app".
try:
    from mcp.server.transport_security import TransportSecuritySettings

    _hosts_env = os.environ.get("RETRIEVAL_ALLOWED_HOSTS", "").strip()
    if _hosts_env:
        _allowed = [h.strip() for h in _hosts_env.split(",") if h.strip()]
        _origins = [f"https://{h}" for h in _allowed] + [f"http://{h}" for h in _allowed]
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_allowed + [f"{h}:*" for h in _allowed],
            allowed_origins=_origins + [f"https://{h}:*" for h in _allowed],
        )
    else:
        # No explicit list -> turn off the host check entirely. Safe here because
        # the platform proxy fronts us and we require a bearer token on every path.
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False,
        )
except Exception as _e:  # older/newer mcp without this module: ignore
    print(f"NOTE: could not set transport_security ({_e}); relying on defaults")


class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        if TOKEN:
            expected = f"Bearer {TOKEN}"
            if request.headers.get("authorization", "") != expected:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


async def healthz(_request):
    return PlainTextResponse("ok")


# streamable-HTTP MCP app, mounted at /mcp by default
app = mcp.streamable_http_app()
app.add_middleware(BearerAuth)
app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))


if __name__ == "__main__":
    if not TOKEN:
        print("WARNING: RETRIEVAL_TOKEN is unset — the endpoint will be UNAUTHENTICATED.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
