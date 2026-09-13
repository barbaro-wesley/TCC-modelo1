"""FastAPI delivery layer for the immutable S10 product service."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hmac
import json
import logging
import os
import threading
import time
from typing import Deque
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .decision import S10DecisionService
from .product import S10ProductService


LOGGER = logging.getLogger("vs_epl_krls.service")


class CostScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    volume_liters: float = Field(default=200_000.0, gt=0, le=50_000_000)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    volume_liters: float = Field(default=200_000.0, gt=0, le=50_000_000)
    # Omitir a UF pede a decisao nacional; informa-la pede a do estado.
    uf: str | None = Field(default=None, min_length=2, max_length=2, pattern=r"^[A-Za-z]{2}$")


class BasisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uf: str = Field(min_length=2, max_length=2, pattern=r"^[A-Za-z]{2}$")
    volume_liters: float = Field(default=200_000.0, gt=0, le=50_000_000)


@dataclass(frozen=True)
class APISettings:
    environment: str = "development"
    api_key: str | None = None
    rate_limit_per_minute: int = 120
    max_request_bytes: int = 32_768
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "testserver")

    @classmethod
    def from_environment(cls) -> "APISettings":
        environment = os.getenv("S10_ENVIRONMENT", "development").strip().lower()
        if environment not in {"development", "test", "production"}:
            raise ValueError("S10_ENVIRONMENT must be development, test, or production")
        rate_limit = int(os.getenv("S10_RATE_LIMIT_PER_MINUTE", "120"))
        max_bytes = int(os.getenv("S10_MAX_REQUEST_BYTES", "32768"))
        if not 1 <= rate_limit <= 10_000:
            raise ValueError("S10_RATE_LIMIT_PER_MINUTE must be between 1 and 10000")
        if not 1024 <= max_bytes <= 1_048_576:
            raise ValueError("S10_MAX_REQUEST_BYTES must be between 1024 and 1048576")
        hosts = tuple(
            host.strip()
            for host in os.getenv(
                "S10_ALLOWED_HOSTS", "127.0.0.1,localhost"
            ).split(",")
            if host.strip()
        )
        return cls(
            environment=environment,
            api_key=os.getenv("S10_API_KEY") or None,
            rate_limit_per_minute=rate_limit,
            max_request_bytes=max_bytes,
            allowed_hosts=hosts,
        )


class _RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = requests_per_minute
        self._clients: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float) -> bool:
        with self._lock:
            if len(self._clients) > 10_000:
                inactive = [key for key, values in self._clients.items() if not values or values[-1] <= now - 60]
                for key in inactive[:2_000]:
                    self._clients.pop(key, None)
            values = self._clients[client]
            while values and values[0] <= now - 60:
                values.popleft()
            if len(values) >= self.limit:
                return False
            values.append(now)
            return True


class _RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.latencies: deque[float] = deque(maxlen=4096)

    def record(self, status_code: int, elapsed: float) -> None:
        with self._lock:
            self.requests += 1
            self.errors += int(status_code >= 500)
            self.latencies.append(elapsed)

    def render(self, ready: bool) -> str:
        with self._lock:
            ordered = sorted(self.latencies)
            p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0
            return "\n".join(
                [
                    "# HELP s10_api_requests_total Total HTTP requests.",
                    "# TYPE s10_api_requests_total counter",
                    f"s10_api_requests_total {self.requests}",
                    "# HELP s10_api_errors_total Total HTTP 5xx responses.",
                    "# TYPE s10_api_errors_total counter",
                    f"s10_api_errors_total {self.errors}",
                    "# HELP s10_api_latency_p95_seconds Recent process-local p95 latency.",
                    "# TYPE s10_api_latency_p95_seconds gauge",
                    f"s10_api_latency_p95_seconds {p95:.9f}",
                    "# HELP s10_product_ready Whether the forecast release is current and verified.",
                    "# TYPE s10_product_ready gauge",
                    f"s10_product_ready {1 if ready else 0}",
                    "",
                ]
            )


def create_app(
    service: S10ProductService,
    *,
    settings: APISettings | None = None,
    decision_service: S10DecisionService | None = None,
) -> FastAPI:
    """Create a read-only API around an already hash-verified release."""

    configuration = settings or APISettings.from_environment()
    if configuration.environment == "production" and not configuration.api_key:
        raise RuntimeError("S10_API_KEY is mandatory in production")
    application = FastAPI(
        title="S10 Intelligence API",
        summary="Weekly Brazilian Diesel B S10 forecasting and cost exposure",
        description=(
            "Read-only, versioned service for the national ANP S10 forecast. "
            "It exposes uncertainty and evidence; it does not execute purchases."
        ),
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(configuration.allowed_hosts),
    )
    limiter = _RateLimiter(configuration.rate_limit_per_minute)
    metrics = _RuntimeMetrics()

    def authorize(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client, time.monotonic()):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate limit exceeded",
                headers={"Retry-After": "60"},
            )
        if configuration.api_key is not None and (
            x_api_key is None
            or not hmac.compare_digest(x_api_key, configuration.api_key)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
            )

    @application.middleware("http")
    async def operational_middleware(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", "")
        if not request_id.isascii() or not request_id.replace("-", "").isalnum() or len(request_id) > 80:
            request_id = uuid4().hex
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > configuration.max_request_bytes
            except ValueError:
                too_large = True
            if too_large:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "request body too large", "request_id": request_id},
                )
                metrics.record(413, time.perf_counter() - started)
                return response
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
        metrics.record(response.status_code, elapsed)
        LOGGER.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(elapsed * 1000, 3),
                },
                separators=(",", ":"),
            )
        )
        return response

    @application.get("/", tags=["service"])
    def root() -> dict[str, object]:
        return {
            "service": "S10 Intelligence API",
            "version": application.version,
            "contract_version": "1.0",
            "read_only": True,
            "operational_status": service.status().as_dict(),
            "resources": {
                "forecast": "/v1/forecast",
                "models": "/v1/models",
                "evidence": "/v1/evidence",
                "cost_scenario": "/v1/scenarios/cost",
                "decision": "/v1/decision",
                "basis": "/v1/basis",
                "governance": "/v1/governance",
                "readiness": "/v1/health/ready",
                "metrics": "/metrics",
                "openapi_contract": "/openapi.json",
            },
            "resource_methods": {
                "forecast": "GET",
                "models": "GET",
                "evidence": "GET",
                "cost_scenario": "POST",
                "decision": "POST",
                "basis": "POST",
                "governance": "GET",
                "readiness": "GET",
                "metrics": "GET",
                "openapi_contract": "GET",
            },
            "request_examples": {
                "cost_scenario": {"volume_liters": 200_000},
                "decision": {"volume_liters": 200_000},
                "basis": {"uf": "RS", "volume_liters": 200_000},
            },
            "usage": {
                "post_requests": "Send POST requests with Content-Type: application/json. "
                "Opening these URLs in a browser sends GET and returns HTTP 405.",
                "expired_forecast": "HTTP 503 on forecast-dependent resources means the "
                "release is unavailable. Check /v1/health/ready for the reason. "
                "An expired forecast requires an updated release from official ANP data.",
            },
        }

    @application.get("/v1/health/live", tags=["health"])
    def live() -> dict[str, object]:
        return {
            "status": "alive",
            "service_version": application.version,
            "started_at_utc": service.started_at_utc,
        }

    @application.get("/v1/health/ready", tags=["health"])
    def ready():
        report = service.status()
        code = 200 if report.serving_ready else 503
        return JSONResponse(status_code=code, content=report.as_dict())

    @application.get("/v1/forecast", tags=["forecast"], dependencies=[Depends(authorize)])
    def forecast() -> dict[str, object]:
        report = service.status()
        if not report.serving_ready:
            raise HTTPException(status_code=503, detail="forecast release is not current")
        return service.forecast()

    @application.post("/v1/scenarios/cost", tags=["decision support"], dependencies=[Depends(authorize)])
    def cost_scenario(payload: CostScenarioRequest) -> dict[str, object]:
        report = service.status()
        if not report.serving_ready:
            raise HTTPException(status_code=503, detail="forecast release is not current")
        return service.cost_scenario(payload.volume_liters).as_dict()

    @application.get("/v1/evidence", tags=["governance"], dependencies=[Depends(authorize)])
    def evidence() -> dict[str, object]:
        return service.model_evidence()

    @application.get("/v1/models", tags=["models"], dependencies=[Depends(authorize)])
    def models() -> dict[str, object]:
        report = service.status()
        if not report.serving_ready:
            raise HTTPException(status_code=503, detail="forecast release is not current")
        return service.model_catalog()

    @application.post("/v1/decision", tags=["decision support"], dependencies=[Depends(authorize)])
    def decision(payload: DecisionRequest) -> dict[str, object]:
        if decision_service is None:
            raise HTTPException(status_code=404, detail="decision layer is not configured")
        report = service.status()
        if not report.serving_ready:
            raise HTTPException(status_code=503, detail="forecast release is not current")
        try:
            return decision_service.decide(payload.volume_liters, uf=payload.uf).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post("/v1/basis", tags=["decision support"], dependencies=[Depends(authorize)])
    def basis(payload: BasisRequest) -> dict[str, object]:
        """Quanto custa orcar pela media nacional em vez da serie do estado."""

        if decision_service is None:
            raise HTTPException(status_code=404, detail="decision layer is not configured")
        try:
            return decision_service.basis(payload.uf, payload.volume_liters).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/v1/governance", tags=["governance"], dependencies=[Depends(authorize)])
    def governance() -> dict[str, object]:
        if decision_service is None:
            raise HTTPException(status_code=404, detail="decision layer is not configured")
        return decision_service.governance()

    @application.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def prometheus_metrics() -> str:
        return metrics.render(service.status().serving_ready)

    return application
