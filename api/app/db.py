from fastapi import Request
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from .config import Settings


def database(settings: Settings):
    engine = create_engine(
        settings.sqlalchemy_url(),
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=0,
        pool_timeout=5,
        connect_args={"connect_timeout": 5, "prepare_threshold": None},
    )
    factory = sessionmaker(engine, expire_on_commit=False)

    @event.listens_for(factory, "after_begin")
    def query_limits(session, transaction, connection):
        # SET LOCAL vale apenas nesta transacao: compativel com pooler do Neon.
        connection.exec_driver_sql(f"SET LOCAL statement_timeout = {settings.query_timeout_ms}")
        connection.exec_driver_sql(f"SET LOCAL lock_timeout = {settings.query_timeout_ms}")

    return engine, factory


def get_db(request: Request):
    with request.app.state.sessions() as session:
        yield session
