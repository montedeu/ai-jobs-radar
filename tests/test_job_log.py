import psycopg
import pytest

from research_radar import collect
from research_radar.job_log import latest_runs, record_run


def test_successful_run_is_logged_with_its_details(database_url):
    with psycopg.connect(database_url) as connection:
        with record_run(connection, "collect") as details:
            details["papers_received"] = 3
        (run,) = latest_runs(connection)
    assert (run.job, run.status, run.details) == ("collect", "succeeded", {"papers_received": 3})
    assert run.finished_at >= run.started_at


def test_failed_run_is_logged_and_the_error_propagates(database_url):
    with psycopg.connect(database_url) as connection:
        with pytest.raises(RuntimeError), record_run(connection, "collect"):
            raise RuntimeError("arXiv is down")
        (run,) = latest_runs(connection)
    assert run.status == "failed"
    assert run.details["error"] == "RuntimeError: arXiv is down"


def test_run_is_visible_as_running_before_it_finishes(database_url):
    with psycopg.connect(database_url) as connection, record_run(connection, "collect"):
        with psycopg.connect(database_url) as observer:
            (run,) = latest_runs(observer)
        assert (run.status, run.finished_at) == ("running", None)


def test_latest_runs_returns_one_row_per_job(database_url):
    with psycopg.connect(database_url) as connection:
        for job in ("collect", "collect", "extract"):
            with record_run(connection, job):
                pass
        runs = latest_runs(connection)
    assert sorted(run.job for run in runs) == ["collect", "extract"]


def test_collect_job_leaves_a_succeeded_run(database_url):
    collect.run(database_url)
    with psycopg.connect(database_url) as connection:
        (run,) = latest_runs(connection)
    assert (run.job, run.status) == ("collect", "succeeded")
