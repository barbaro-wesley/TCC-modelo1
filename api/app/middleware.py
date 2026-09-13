import json
import logging
import time
from uuid import uuid4

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("s10.access")


class Guardrails:
    """ASGI: valida tamanho real, inclusive requests sem Content-Length."""

    def __init__(self, app, settings, limiter):
        self.app, self.settings, self.limiter = app, settings, limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        request_id = str(uuid4())  # nao confiar em IDs enviados pelo consumidor
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        code = 500

        async def send_headers(message):
            nonlocal code
            if message["type"] == "http.response.start":
                code = message["status"]
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"
                ]
                headers.extend(
                    [
                        (b"cache-control", b"no-store"),
                        (b"x-request-id", request_id.encode()),
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"x-frame-options", b"DENY"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        try:
            # Liveness e a unica rota que nao depende do Redis.
            if request.url.path != "/health/live":
                await run_in_threadpool(
                    self.limiter.check, "global", "all", self.settings.global_rate_per_minute
                )
                ip = request.client.host if request.client else "unknown"
                await run_in_threadpool(
                    self.limiter.check, "ip", ip, self.settings.ip_rate_per_minute
                )
                if request.url.path.startswith("/api/v1/auth/"):
                    await run_in_threadpool(
                        self.limiter.check, "auth-ip", ip, self.settings.login_rate_per_minute
                    )
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    length = int(length)
                except ValueError:
                    raise HTTPException(400, detail={"code": "invalid_content_length"}) from None
                if length < 0:
                    raise HTTPException(400, detail={"code": "invalid_content_length"})
                if length > self.settings.max_body_bytes:
                    raise HTTPException(413, detail={"code": "body_too_large"})
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > self.settings.max_body_bytes:
                    raise HTTPException(413, detail={"code": "body_too_large"})
                if not message.get("more_body", False):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if delivered:
                    return await receive()
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self.app(scope, bounded_receive, send_headers)
        except HTTPException as exc:
            await JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )(scope, receive, send_headers)
        finally:
            logger.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": code,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    }
                )
            )
