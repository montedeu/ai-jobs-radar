from alembic import context
from sqlalchemy import create_engine, make_url, pool


def run_migrations() -> None:
    # SQLAlchemy is only Alembic's engine here; application code uses psycopg directly.
    url = make_url(context.config.attributes["database_url"]).set(drivername="postgresql+psycopg")
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


run_migrations()
