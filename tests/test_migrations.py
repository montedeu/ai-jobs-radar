import psycopg

from research_radar import migrate


def test_migrations_apply_and_roll_back_cleanly(database_url):
    migrate.downgrade(database_url, "base")
    with psycopg.connect(database_url) as connection:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    assert tables == [("alembic_version",)]
    migrate.upgrade(database_url)
