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


def test_every_host_of_a_multi_host_address_is_rewritten():
    url = "postgresql://u:p@ep-a-1-pooler.x.neon.tech,ep-b-2-pooler.x.neon.tech/db"

    assert conninfo_to_dict(direct_url(url))["host"] == "ep-a-1.x.neon.tech,ep-b-2.x.neon.tech"


def test_endpoint_named_in_options_is_rewritten_too():
    url = "postgresql://u:p@ep-a-1-pooler.x.neon.tech/db?options=endpoint%3Dep-a-1-pooler"

    assert conninfo_to_dict(direct_url(url))["options"] == "endpoint=ep-a-1"


def test_upper_case_pooled_host_is_recognised():
    url = "postgresql://u:p@EP-A-1-POOLER.x.neon.tech/db"

    assert conninfo_to_dict(direct_url(url))["host"] == "EP-A-1.x.neon.tech"


def test_pooler_pinned_by_address_is_refused():
    url = "host=ep-a-1-pooler.x.neon.tech hostaddr=203.0.113.7 dbname=db"

    with pytest.raises(ValueError, match="hostaddr"):
        direct_url(url)


def test_direct_host_with_an_address_is_left_alone():
    url = "host=ep-a-1.x.neon.tech hostaddr=203.0.113.7 dbname=db"

    assert conninfo_to_dict(direct_url(url)) == conninfo_to_dict(url)
