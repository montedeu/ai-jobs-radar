"""Applies the schema migrations in ``research_radar/migrations``."""

from pathlib import Path

from alembic import command
from alembic.config import Config

from research_radar.database import database_url

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    # Passed as an attribute, not as the sqlalchemy.url option: options go through
    # configparser, which would read a "%" in the password as interpolation.
    config.attributes["database_url"] = url or database_url()
    return config


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)
