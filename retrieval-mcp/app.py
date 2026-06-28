"""
HTTP entrypoint for RetriEval — deploy this to reach the server from anywhere.

Bearer-token auth + DNS-rebinding fix so the streamable-HTTP MCP works behind a
public host (Railway/Render/Fly). Clients connect to https://<host>/mcp with
header  Authorization: Bearer <RETRIEVAL_TOKEN>.  Health check at /healthz.
"""
from __future__ import annotations
import os

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

# 1) Relax FastMCP's DNS-rebinding host check BEFORE the app/session-manager is
#    built (it reads this setting once, lazily, on first streamable_http_app()).
from server import mcp  # noqa: E402
from server import run_sandbox_eval  # noqa: E402
from judge import sandbox_models  # noqa: E402

try:
    from mcp.server.transport_security import TransportSecuritySettings
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
except Exception as _e:
    print(f"NOTE: could not set transport_security ({_e})")

TOKEN = os.environ.get("RETRIEVAL_TOKEN", "")
SANDBOX_SECRET = os.environ.get("SANDBOX_SECRET", "")
PUBLIC_PATHS = ("/healthz", "/sandbox", "/sandbox/models")


# 2) Belt-and-suspenders: normalize Host/Origin so the rebinding check (if any
#    layer still runs it) always sees a trusted localhost value. The public
#    proxy already terminates TLS; we enforce our own bearer token below.
class NormalizeHost(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        headers = request.scope.get("headers")
        if headers:
            new = []
            for k, v in headers:
                lk = k.decode().lower() if isinstance(k, bytes) else str(k).lower()
                if lk == b"host".decode() or lk == "host":
                    new.append((b"host", b"localhost"))
                elif lk == "origin":
                    new.append((b"origin", b"http://localhost"))
                else:
                    new.append((k, v))
            request.scope["headers"] = new
        return await call_next(request)


class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        if TOKEN:
            if request.headers.get("authorization", "") != f"Bearer {TOKEN}":
                return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


async def healthz(_request):
    return PlainTextResponse("ok")


async def sandbox_models_route(_request):
    return JSONResponse({"models": sandbox_models()})


async def sandbox_route(request):
    # server-to-server only: the Vercel /api/sandbox holds this secret and does
    # the per-IP rate limiting. Reject direct public hits.
    if not SANDBOX_SECRET or request.headers.get("x-sandbox-secret", "") != SANDBOX_SECRET:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad_json"}, status_code=400)
    cases = body.get("cases")
    metric = body.get("metric", "faithfulness")
    model = body.get("model", "")
    result = run_sandbox_eval(cases or [], metric, model)
    if isinstance(result, dict) and result.get("error") == "budget_exceeded":
        return JSONResponse(result, status_code=429)
    if isinstance(result, dict) and result.get("error"):
        return JSONResponse(result, status_code=400)
    return JSONResponse(result)


app = mcp.streamable_http_app()
# order matters: auth runs first (outermost), then host normalization
app.add_middleware(NormalizeHost)
app.add_middleware(BearerAuth)
app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
app.router.routes.append(Route("/sandbox", sandbox_route, methods=["POST"]))
app.router.routes.append(Route("/sandbox/models", sandbox_models_route, methods=["GET"]))


if __name__ == "__main__":
    if not TOKEN:
        print("WARNING: RETRIEVAL_TOKEN is unset — the endpoint will be UNAUTHENTICATED.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
