from concurrent.futures import ThreadPoolExecutor

import pytest
from alembic import command
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from api.app.models import AuthSession, MonthlyUsage, Plan, UsageEvent
from api.tests.conftest import PASSWORD, onboard


def test_migration_and_database_timeout(backend):
    client, admin, factory, settings, alembic_config = backend
    assert client.get("/health/ready").status_code == 200
    command.check(alembic_config)
    with factory() as db:
        assert db.scalar(text("SHOW statement_timeout")) == "5s"
        db.execute(text("SET LOCAL statement_timeout = 30"))
        with pytest.raises(DBAPIError):
            db.execute(text("SELECT pg_sleep(0.2)"))


def test_auth_rotation_reuse_logout_and_reset(backend):
    client, admin, factory, *_ = backend
    org_id, headers, tokens, _ = onboard(client, admin)
    assert client.get("/api/v1/me").status_code == 401
    assert client.get("/api/v1/me", headers=headers).status_code == 200
    refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    new_headers = {"Authorization": "Bearer " + refreshed.json()["access_token"]}
    assert client.get("/api/v1/me", headers=headers).status_code == 401
    assert client.get("/api/v1/me", headers=new_headers).status_code == 200
    assert (
        client.post(
            "/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}
        ).status_code
        == 401
    )
    assert client.get("/api/v1/me", headers=new_headers).status_code == 401
    login = client.post(
        "/api/v1/auth/login", json={"email": "alpha@example.com", "password": PASSWORD}
    ).json()
    current = {"Authorization": "Bearer " + login["access_token"]}
    user_id = client.get("/api/v1/me", headers=current).json()["id"]
    reset = client.post(f"/api/v1/admin/users/{user_id}/password-reset", headers=admin)
    assert reset.status_code == 200
    body = {"token": reset.json()["token"], "password": "New-password-2026!"}
    assert client.post("/api/v1/auth/reset-password", json=body).status_code == 204
    assert client.post("/api/v1/auth/reset-password", json=body).status_code == 401
    assert client.get("/api/v1/me", headers=current).status_code == 401
    assert (
        client.post(
            "/api/v1/auth/login", json={"email": "alpha@example.com", "password": PASSWORD}
        ).status_code
        == 401
    )
    new_login = client.post(
        "/api/v1/auth/login", json={"email": "alpha@example.com", "password": body["password"]}
    ).json()
    new_headers = {"Authorization": "Bearer " + new_login["access_token"]}
    assert client.post("/api/v1/auth/logout", headers=new_headers).status_code == 204
    assert client.get("/api/v1/me", headers=new_headers).status_code == 401
    with factory() as db:
        assert tokens["refresh_token"] not in str(
            db.scalars(select(AuthSession.refresh_hash)).all()
        )


def test_tenant_isolation_roles_pagination_and_seats(backend):
    client, admin, *_ = backend
    _, alpha, _, _ = onboard(client, admin, seats=2)
    _, beta, _, _ = onboard(client, admin, other=True)
    beta_id = client.get("/api/v1/me", headers=beta).json()["id"]
    assert (
        client.patch(
            f"/api/v1/organization/users/{beta_id}", headers=alpha, json={"active": False}
        ).status_code
        == 404
    )
    assert client.get("/api/v1/admin/organizations", headers=alpha).status_code == 403
    response = client.get("/api/v1/organization/users?page_size=1", headers=alpha).json()
    assert response["total"] == 1 and response["items"][0]["email"] == "alpha@example.com"
    assert "password_hash" not in response["items"][0]
    for route in (
        "plans",
        "organization/users",
        "organization/api-keys",
        "usage/events",
        "history",
    ):
        assert client.get(f"/api/v1/{route}?page_size=101", headers=alpha).status_code == 422
    viewer = client.post(
        "/api/v1/organization/users",
        headers=alpha,
        json={
            "name": "Viewer",
            "email": "viewer@example.com",
            "password": PASSWORD,
            "role": "viewer",
        },
    )
    assert viewer.status_code == 201, viewer.text
    assert (
        client.post(
            "/api/v1/organization/users",
            headers=alpha,
            json={"name": "Extra", "email": "extra@example.com", "password": PASSWORD},
        ).status_code
        == 409
    )
    login = client.post(
        "/api/v1/auth/login", json={"email": "viewer@example.com", "password": PASSWORD}
    ).json()
    view = {"Authorization": "Bearer " + login["access_token"]}
    assert (
        client.post("/api/v1/scenarios/cost", headers=view, json={"volume_liters": 10}).status_code
        == 403
    )
    assert client.get("/api/v1/organization/users", headers=view).status_code == 403
    alpha_id = client.get("/api/v1/me", headers=alpha).json()["id"]
    assert (
        client.patch(
            f"/api/v1/organization/users/{alpha_id}", headers=alpha, json={"active": False}
        ).status_code
        == 409
    )


def test_quota_is_atomic_and_survives_redis_loss(backend, monkeypatch):
    client, admin, factory, settings, _ = backend
    org_id, headers, _, _ = onboard(client, admin, quota=3)

    def call(_):
        return client.get("/api/v1/forecast", headers=headers).status_code

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(call, range(12)))
    assert codes.count(200) == 3, codes
    assert codes.count(429) == 9, codes
    usage = client.get("/api/v1/usage", headers=headers).json()
    assert usage["units"] == 3 and usage["remaining"] == 0
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(UsageEvent)) == 3
        assert db.scalar(select(MonthlyUsage.units)) == 3
    # Perder os baldes Redis nao devolve unidades comerciais.
    redis = client.app.state.limiter.redis
    for key in redis.scan_iter(match=f"{settings.redis_prefix}:*"):
        redis.delete(key)
    assert client.get("/api/v1/forecast", headers=headers).status_code == 429
    assert client.get("/api/v1/usage/events?page_size=2", headers=headers).json()["total"] == 3
    # Mudar o periodo abre outra linha, sem apagar o consumo anterior.
    from datetime import timedelta

    from api.app import product
    from api.app.models import now

    next_month = (now().replace(day=1) + timedelta(days=32)).replace(day=1)
    monkeypatch.setattr(product, "now", lambda: next_month)
    new_period = client.get("/api/v1/history", headers=headers)
    assert new_period.status_code == 200
    assert new_period.headers["X-Quota-Period"] == next_month.strftime("%Y-%m")
    with factory() as db:
        assert sorted(db.scalars(select(MonthlyUsage.units)).all()) == [1, 3]


def test_failed_forecast_not_metered_and_history_bounded(backend):
    client, admin, _, settings, _ = backend
    _, headers, _, _ = onboard(client, admin)
    history = client.get("/api/v1/history?page=2&page_size=5", headers=headers)
    assert history.status_code == 200 and len(history.json()["items"]) == 5
    assert history.json()["total"] == 30
    assert client.get("/api/v1/status", headers=headers).json().get("erro") is None
    (settings.data_dir / "previsao.json").write_text("{}", encoding="utf-8")
    assert client.get("/api/v1/forecast", headers=headers).status_code == 503
    assert client.get("/api/v1/usage", headers=headers).json()["units"] == 1


def test_api_keys_revocation_subscription_and_plan_snapshot(backend):
    client, admin, factory, *_ = backend
    org_id, headers, _, sub = onboard(client, admin)
    created = client.post("/api/v1/organization/api-keys", headers=headers, json={"label": "ERP"})
    assert created.status_code == 201, created.text
    key_headers = {"X-API-Key": created.json()["api_key"]}
    assert client.get("/api/v1/forecast", headers=key_headers).status_code == 200
    assert client.get("/api/v1/organization/users", headers=key_headers).status_code == 403
    assert (
        "api_key"
        not in client.get("/api/v1/organization/api-keys", headers=headers).json()["items"][0]
    )
    assert (
        client.delete(
            f"/api/v1/organization/api-keys/{created.json()['id']}", headers=headers
        ).status_code
        == 204
    )
    assert client.get("/api/v1/forecast", headers=key_headers).status_code == 401
    with factory() as db:
        from uuid import UUID

        plan = db.get(Plan, UUID(sub["plan_id"]))
        plan.price_cents = 99900
        db.commit()
    assert client.get("/api/v1/subscription", headers=headers).json()["price_cents"] == 29900
    body = {name: sub[name] for name in ("plan_id", "status", "starts_at", "ends_at")}
    body["status"] = "suspended"
    assert (
        client.put(
            f"/api/v1/admin/organizations/{org_id}/subscription", headers=admin, json=body
        ).status_code
        == 200
    )
    assert client.get("/api/v1/forecast", headers=headers).status_code == 403
    assert client.get("/api/v1/usage", headers=headers).status_code == 200
    assert (
        client.patch(
            f"/api/v1/admin/organizations/{org_id}", headers=admin, json={"active": False}
        ).status_code
        == 200
    )
    assert client.get("/api/v1/me", headers=headers).status_code == 401


def test_rate_limit_aggregates_users_and_keys_not_ip(backend):
    client, admin, *_ = backend
    _, alpha, _, _ = onboard(client, admin, rate=2)
    _, beta, _, _ = onboard(client, admin, other=True, rate=2)
    created = client.post(
        "/api/v1/organization/api-keys", headers=alpha, json={"label": "ERP"}
    ).json()
    assert (
        client.get("/api/v1/forecast", headers={"X-API-Key": created["api_key"]}).status_code == 200
    )
    assert client.get("/api/v1/forecast", headers=alpha).status_code == 429
    assert client.get("/api/v1/forecast", headers=beta).status_code == 200
