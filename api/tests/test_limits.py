from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from redis.exceptions import ConnectionError

from api.app.limits import RateLimiter
from api.app.main import create_app
from api.app.pagination import Page, pagination
from api.tests.conftest import settings


def test_atomic_limit_shared_between_workers(redis_client):
    server = fakeredis.FakeServer()
    workers = [RateLimiter(fakeredis.FakeRedis(server=server), "test") for _ in range(2)]

    def attempt(i):
        try:
            workers[i % 2].check("tenant", "alpha", 7)
            return True
        except HTTPException as exc:
            assert exc.status_code == 429
            assert int(exc.headers["Retry-After"]) > 0
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        assert sum(pool.map(attempt, range(30))) == 7
    assert workers[0].check("tenant", "beta", 7)["RateLimit-Remaining"] == "6"


def test_redis_failure_fails_closed(redis_client, monkeypatch):
    limiter = RateLimiter(redis_client, "test")

    def fail(*args, **kwargs):
        raise ConnectionError("secret redis hostname")

    monkeypatch.setattr(limiter, "script", fail)
    with pytest.raises(HTTPException) as error:
        limiter.check("global", "all", 10)
    assert error.value.status_code == 503
    assert "secret" not in str(error.value.detail)


def test_global_limit_includes_unauthenticated_requests(redis_client):
    app = create_app(
        settings(global_rate_per_minute=2), session_factory=lambda: None, redis_client=redis_client
    )
    with TestClient(app) as client:
        assert client.get("/missing").status_code == 404
        assert client.get("/missing").status_code == 404
        denied = client.get("/missing")
        assert denied.status_code == 429
        assert denied.json()["detail"]["scope"] == "global"
        assert client.get("/health/live").status_code == 200


def test_body_limit_rejects_chunked_input(app_no_db):
    with TestClient(app_no_db) as client:
        response = client.post("/missing", content=iter([b"x" * 20_000, b"y" * 20_000]))
        assert response.status_code == 413
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Request-ID"]


def test_pagination_caps_are_global(app_no_db):
    from starlette.requests import Request

    request = Request({"type": "http", "app": app_no_db})
    assert pagination(request, 1, None) == Page(1, 20)
    assert pagination(request, 2, 100).offset == 100
    for page, size in ((1, 101), (102, 100)):
        with pytest.raises(HTTPException) as error:
            pagination(request, page, size)
        assert error.value.status_code == 422


def test_configuration_rejects_unsafe_production():
    with pytest.raises(ValidationError):
        settings(environment="production")  # TLS required
    with pytest.raises(ValidationError):
        settings(database_url="sqlite:///tenancy.db")
    with pytest.raises(ValidationError):
        settings(cors_origins=["*"])


def test_all_business_routes_reject_anonymous_access(app_no_db):
    from api.app.db import get_db

    app_no_db.dependency_overrides[get_db] = lambda: None
    with TestClient(app_no_db) as client:
        for path in (
            "me",
            "forecast",
            "history",
            "status",
            "plans",
            "subscription",
            "usage",
            "usage/events",
            "organization/users",
            "organization/api-keys",
            "admin/users",
            "admin/organizations",
            "admin/audit-logs",
        ):
            assert client.get(f"/api/v1/{path}").status_code == 401, path
        assert client.get("/api/previsao").status_code == 404
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_production_schema_requires_admin(redis_client):
    from api.app.db import get_db

    config = settings(
        environment="production", database_url="postgresql://unused@localhost/test?sslmode=require"
    )
    app = create_app(config, session_factory=lambda: None, redis_client=redis_client)
    app.dependency_overrides[get_db] = lambda: None
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 401


def test_validation_does_not_echo_passwords(app_no_db):
    from api.app.db import get_db

    app_no_db.dependency_overrides[get_db] = lambda: None
    with TestClient(app_no_db) as client:
        result = client.post(
            "/api/v1/auth/login", json={"email": "invalid", "password": "sensitive-value"}
        )
        assert result.status_code == 422
        assert "sensitive-value" not in result.text


def test_password_whitespace_and_cnpj():
    from api.app.schemas import Login, OrganizationCreate

    assert Login(email="test@example.com", password="  password  ").password == "  password  "
    owner = {"name": "Owner", "email": "owner@example.com", "password": "a-test-password!"}
    for cnpj in ("04.252.011/0001-10", "12.ABC.345/01DE-35"):
        assert OrganizationCreate(name="Organization", cnpj=cnpj, owner=owner).cnpj
    with pytest.raises(ValidationError):
        OrganizationCreate(name="Organization", cnpj="00.000.000/0000-00", owner=owner)
