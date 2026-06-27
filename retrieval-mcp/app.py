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
