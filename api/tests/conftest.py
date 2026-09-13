import json
import os
from datetime import timedelta
from uuid import uuid4

import fakeredis
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from api.app.config import Settings
from api.app.db import database
from api.app.main import create_app
from api.app.models import User, now
from api.app.security import passwords

PASSWORD = "Test-password-2026!"


def settings(**overrides):
    return Settings(
        _env_file=None,
        **{
            "environment": "test",
            "database_url": "postgresql://unused:unused@localhost/unused",
            "redis_url": "redis://localhost/0",
            "jwt_secret": "test-only-secret-01234567890123456789",
            "global_rate_per_minute": 100_000,
            "ip_rate_per_minute": 100_000,
            "login_rate_per_minute": 100_000,
            **overrides,
        },
    )


@pytest.fixture
def redis_client():
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def app_no_db(redis_client):
    return create_app(settings(), session_factory=lambda: None, redis_client=redis_client)


@pytest.fixture
def backend(tmp_path):
    """PostgreSQL real, migracao real e schema novo por teste; nunca limpa banco existente."""
    raw_url = os.getenv("S10_TEST_DATABASE_URL")
    if not raw_url:
        pytest.skip("Defina S10_TEST_DATABASE_URL para executar integracao PostgreSQL real")
    schema = "s10_test_" + uuid4().hex
    base_url = make_url(raw_url).set(drivername="postgresql+psycopg")
    control = create_engine(
        base_url, connect_args={"connect_timeout": 5, "prepare_threshold": None}
    )
    with control.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = base_url.update_query_dict({"options": f"-csearch_path={schema}"})
    config = settings(
        database_url=url.render_as_string(hide_password=False),
        data_dir=tmp_path,
        redis_prefix=schema,
        pool_size=12,
    )
    alembic_config = Config("api/alembic.ini")
    alembic_config.attributes["settings"] = config
    engine = None
    redis = None
    try:
        command.upgrade(alembic_config, "head")
        engine, factory = database(config)
        redis_url = os.getenv("S10_TEST_REDIS_URL")
        if redis_url:
            from redis import Redis

            redis = Redis.from_url(redis_url, decode_responses=True)
        else:
            redis = fakeredis.FakeRedis(decode_responses=True)
        timestamp = now()
        forecast = {
            "ultima_semana_observada": timestamp.date().isoformat(),
            "semana_prevista": (timestamp + timedelta(days=7)).date().isoformat(),
            "previsao_pontual": 6.9,
            "preco_observado_ultima_semana": 6.8,
            "p10": 6.7,
            "p90": 7.1,
            "modelo": "test",
        }
        (tmp_path / "previsao.json").write_text(json.dumps(forecast), encoding="utf-8")
        (tmp_path / "historico.json").write_text(
            json.dumps(
                {
                    "serie": [
                        {
                            "data": (timestamp - timedelta(days=n * 7)).date().isoformat(),
                            "revenda": 6.8,
                        }
                        for n in range(30)
                    ],
                    "previsoes": [],
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / "status.json").write_text(
            json.dumps({"status": "atualizado", "erro": "private"}), encoding="utf-8"
        )
        with factory() as db:
            db.add(
                User(
                    name="Admin",
                    email="admin@example.com",
                    role="platform_admin",
                    password_hash=passwords.hash(PASSWORD),
                )
            )
            db.commit()
        app = create_app(config, session_factory=factory, redis_client=redis)
        with TestClient(app) as client:
            result = client.post(
                "/api/v1/auth/login", json={"email": "admin@example.com", "password": PASSWORD}
            )
            assert result.status_code == 200, result.text
            admin = {"Authorization": "Bearer " + result.json()["access_token"]}
            yield client, admin, factory, config, alembic_config
    finally:
        if engine:
            engine.dispose()
        # Limpa somente as chaves deste teste, sem FLUSHDB.
        if redis:
            for key in redis.scan_iter(match=f"{schema}:*"):
                redis.delete(key)
            redis.close()
        with control.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        control.dispose()


def onboard(client, admin, *, other=False, quota=100, rate=1000, seats=5, api_access=True):
    email = "beta@example.com" if other else "alpha@example.com"
    org = client.post(
        "/api/v1/admin/organizations",
        headers=admin,
        json={
            "name": "Beta" if other else "Alpha",
            "cnpj": "11.222.333/0001-81" if other else "04.252.011/0001-10",
            "owner": {"name": "Owner", "email": email, "password": PASSWORD},
        },
    )
    assert org.status_code == 201, org.text
    org_id = org.json()["id"]
    plan = client.post(
        "/api/v1/admin/plans",
        headers=admin,
        json={
            "name": "Pilot",
            "price_cents": 29900,
            "monthly_quota": quota,
            "rate_per_minute": rate,
            "max_users": seats,
            "api_access": api_access,
        },
    )
    assert plan.status_code == 201, plan.text
    current = now()
    sub = client.put(
        f"/api/v1/admin/organizations/{org_id}/subscription",
        headers=admin,
        json={
            "plan_id": plan.json()["id"],
            "status": "active",
            "starts_at": (current - timedelta(hours=1)).isoformat(),
            "ends_at": (current + timedelta(days=30)).isoformat(),
        },
    )
    assert sub.status_code == 200, sub.text
    login = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    return org_id, headers, login.json(), sub.json()
