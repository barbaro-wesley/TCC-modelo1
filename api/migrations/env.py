from alembic import context
from api.app.config import Settings
from api.app.models import Base
from sqlalchemy import create_engine, pool

settings = context.config.attributes.get("settings") or Settings()
target_metadata = Base.metadata

if context.is_offline_mode():
    context.configure(
        url=settings.sqlalchemy_url(migration=True),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(
        settings.sqlalchemy_url(migration=True),
        poolclass=pool.NullPool,
        connect_args={"connect_timeout": 10, "prepare_threshold": None},
    )
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
