"""Integration with real PostgreSQL: atomic visibility, audit and concurrent writers."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from threading import Event

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from forecast_store.repository import Publisher, current_forecast, history_page
from forecast_store.schema import forecasts, metrics, observations, publication, runs

from .conftest import onboard

PROTOCOL = "calendar-mature-labels-v1"


def payload(origin=None, point=6.9):
    origin = origin or date.today()
    forecast = {
        "temporal_protocol": PROTOCOL,
        "modelo": "test",
        "ultima_semana_observada": origin.isoformat(),
        "semana_prevista": (origin + timedelta(days=7)).isoformat(),
        "previsao_pontual": point,
        "preco_observado_ultima_semana": 6.8,
        "p10": 6.7,
        "p90": 7.1,
    }
    series = [
        {"data": (origin - timedelta(days=7 * n)).isoformat(), "revenda": 6.8} for n in range(30)
    ]
    return forecast, series, [{"model": "test", "horizon": 1, "rmse": 0.07}]


def publisher(backend):
    return Publisher(backend[2].kw["bind"])


def start(pub):
    return pub.start(code_version="test", protocol=PROTOCOL, config={})


def test_failed_publication_rolls_back_all_results(backend):
    pub = publisher(backend)
    with pub.engine.connect() as db:
        previous = current_forecast(db)
    run_id = start(pub)
    forecast, series, scores = payload()
    with pytest.raises(IntegrityError):
        pub.publish(run_id, forecast, series, scores * 2)
    pub.finish(run_id, "failed", error_code="IntegrityError")
    with pub.engine.connect() as db:
        assert current_forecast(db) == previous
        for table in (forecasts, observations, metrics):
            assert (
                db.scalar(
                    sa.select(sa.func.count()).select_from(table).where(table.c.run_id == run_id)
                )
                == 0
            )
        assert db.scalar(sa.select(runs.c.status).where(runs.c.id == run_id)) == "failed"
    client, admin, *_ = backend
    _, headers, *_ = onboard(client, admin)
    assert client.get("/api/v1/forecast", headers=headers).status_code == 200
    assert client.get("/api/v1/status", headers=headers).json()["status"] == "erro"


def test_reader_sees_old_publication_until_commit(backend):
    pub = publisher(backend)
    run_id = start(pub)
    entered, release = Event(), Event()

    def before_cursor(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE model_publication"):
            entered.set()
            assert release.wait(10), "reader did not complete"

    sa.event.listen(pub.engine, "before_cursor_execute", before_cursor)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pub.publish, run_id, *payload(point=7.0))
            try:
                assert entered.wait(10), "writer did not reach publication"
                with pub.engine.connect() as db:
                    assert current_forecast(db)["previsao_pontual"] == 6.9
                    assert history_page(db, "observed", None, None, 0, 10)[1] == 30
            finally:
                release.set()
            assert future.result(timeout=10) == "published"
    finally:
        sa.event.remove(pub.engine, "before_cursor_execute", before_cursor)
    with pub.engine.connect() as db:
        assert current_forecast(db)["run_id"] == run_id


def test_retry_reprocessing_and_late_writer_preserve_history(backend):
    pub = publisher(backend)
    older, newer = start(pub), start(pub)
    assert pub.publish(newer, *payload(point=7.0)) == "published"
    assert pub.publish(newer, *payload(point=7.0)) == "published"
    assert pub.publish(older, *payload()) == "superseded"
    regressed = start(pub)
    assert pub.publish(regressed, *payload(date.today() - timedelta(days=7))) == "superseded"
    with pub.engine.connect() as db:
        assert current_forecast(db)["run_id"] == newer
        rows, total = history_page(db, "forecasts", None, None, 0, 10)
        assert total == 1 and rows[0]["run_id"] == newer
        assert db.scalar(sa.select(sa.func.count()).select_from(forecasts)) == 2
    next_run = start(pub)
    pub.publish(next_run, *payload(date.today() + timedelta(days=7)))
    with pub.engine.connect() as db:
        rows, total = history_page(db, "forecasts", None, None, 0, 10)
        assert total == 2
        assert rows[0]["preco_realizado"] is None
        assert rows[1]["preco_realizado"] == 6.8
        assert rows[1]["erro"] == -0.2
        assert db.scalar(sa.select(forecasts.c.point).where(forecasts.c.run_id == newer)) == 7.0
        filtered, count = history_page(
            db,
            "forecasts",
            date.today() + timedelta(days=7),
            date.today() + timedelta(days=7),
            0,
            1,
        )
        assert count == 1 and filtered[0]["run_id"] == newer


def test_empty_database_is_ready_but_product_not_billable(backend):
    client, admin, factory, *_ = backend
    with factory.begin() as db:
        db.execute(publication.update().values(run_id=None))
    _, headers, *_ = onboard(client, admin)
    assert client.get("/health/ready").status_code == 200
    assert client.get("/api/v1/forecast", headers=headers).status_code == 503
    assert client.get("/api/v1/history", headers=headers).status_code == 503
    assert client.get("/api/v1/usage", headers=headers).json()["units"] == 0
    with factory.begin() as db:
        db.execute(sa.text("UPDATE alembic_version SET version_num = '0001_platform'"))
    assert client.get("/health/ready").status_code == 503
