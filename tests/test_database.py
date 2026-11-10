import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from research_radar.database import direct_url

POOLED = "postgresql://user:p%40ss@ep-cool-name-123456-pooler.eu-central-1.aws.neon.tech/db?sslmode=require"


def test_pooled_neon_address_becomes_the_direct_one():
    parts = conninfo_to_dict(direct_url(POOLED))

    assert parts["host"] == "ep-cool-name-123456.eu-central-1.aws.neon.tech"


def test_everything_but_the_host_is_kept():
    before, after = conninfo_to_dict(POOLED), conninfo_to_dict(direct_url(POOLED))

    del before["host"], after["host"]
    assert before == after
    assert after["password"] == "p@ss"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://user:pw@ep-cool-name-123456.eu-central-1.aws.neon.tech/db",
        "postgresql://user:pw@localhost:5432/db",
        "postgresql://user:pw@my-pooler-db.example.com/db",
    ],
)
def test_address_that_is_not_a_neon_pooler_is_left_alone(url):
    assert conninfo_to_dict(direct_url(url)) == conninfo_to_dict(url)


def test_result_is_usable_by_psycopg_for_a_socket_url(postgres_url):
    with psycopg.connect(direct_url(postgres_url)) as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)
