from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import auth, management, product
from .config import Settings
from .db import database
from .dependencies import admin
from .limits import RateLimiter
from .middleware import Guardrails


def create_app(settings=None, *, session_factory=None, redis_client=None):
    settings = settings or Settings()
    engine = None
    if session_factory is None:
        engine, session_factory = database(settings)
    redis_client = (
        redis_client
        if redis_client is not None
        else Redis.from_url(
            settings.redis_url.get_secret_value(),
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
            max_connections=50,
        )
    )
    limiter = RateLimiter(redis_client, settings.redis_prefix)

    @asynccontextmanager
    async def lifespan(app):
        yield
        redis_client.close()
        if engine is not None:
            engine.dispose()

    app = FastAPI(
        title="Diesel S10 Platform API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.sessions = session_factory
    app.state.limiter = limiter
    app.include_router(auth.router)
    app.include_router(management.router)
    app.include_router(product.router)

    @app.get(
        "/openapi.json",
        include_in_schema=False,
        dependencies=[Depends(admin)] if settings.environment == "production" else [],
    )
    def openapi():
        return app.openapi()

    # FastAPI docs need an explicit URL because the schema itself is access-controlled.
    if settings.environment != "production":
        from fastapi.openapi.docs import get_swagger_ui_html

        @app.get("/docs", include_in_schema=False)
        def docs():
            return get_swagger_ui_html(openapi_url="/openapi.json", title="S10 API")

    @app.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive"}

    @app.get("/health/ready", tags=["health"])
    def ready():
        try:
            with session_factory() as db:
                revision = db.scalar(text("SELECT version_num FROM alembic_version LIMIT 1"))
                if revision != "0002_model_publication":
                    return JSONResponse({"status": "migration_required"}, status_code=503)
                if db.scalar(text("SELECT id FROM model_publication WHERE id = 1")) != 1:
                    return JSONResponse({"status": "migration_required"}, status_code=503)
            redis_client.ping()
        except (SQLAlchemyError, RedisError):
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {"status": "ready"}

    @app.exception_handler(IntegrityError)
    async def conflict(request: Request, exc):
        return JSONResponse({"detail": {"code": "conflicting_record"}}, status_code=409)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc):
        return JSONResponse(
            {"detail": {"code": "database_unavailable_or_query_timeout"}}, status_code=503
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        # Pydantic inclui o input original nos erros: poderia refletir senha/token.
        errors = [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    app.add_middleware(Guardrails, settings=settings, limiter=limiter)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=[
            "X-Request-ID",
            "RateLimit-Limit",
            "RateLimit-Remaining",
            "RateLimit-Reset",
            "X-RateLimit-Scope",
            "Retry-After",
            "X-Quota-Limit",
            "X-Quota-Remaining",
            "X-Quota-Period",
        ],
    )
    return app
