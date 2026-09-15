"""Credencial exclusiva do publicador; nao importa configuracoes da API."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from forecast_store.repository import Publisher


def publisher_from_environment():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env.training", override=False)
    raw_url = os.environ.get("S10_TRAINING_DATABASE_URL")
    if not raw_url:
        raise ValueError("Defina S10_TRAINING_DATABASE_URL para o treinamento")
    url = make_url(raw_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("O publicador requer PostgreSQL")
    if os.getenv("S10_ENVIRONMENT", "production") == "production" and url.query.get(
        "sslmode"
    ) not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("O banco de producao requer sslmode=require ou verificacao de certificado")
    engine = create_engine(
        url.set(drivername="postgresql+psycopg"),
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        connect_args={"connect_timeout": 10, "prepare_threshold": None},
    )
    return Publisher(engine)
