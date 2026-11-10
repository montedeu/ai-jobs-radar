from datetime import UTC, date, datetime, timedelta

import psycopg
import pytest

from research_radar import collect, papers
from research_radar.arxiv_api import Paper
from research_radar.job_log import latest_runs
from research_radar.settings import HISTORY_START

NOW = datetime.now(UTC).replace(microsecond=0)


def paper(arxiv_id, *, updated=NOW, first=None, version=1, abstract="An abstract.", **fields):
    return Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=f"Title of {arxiv_id}",
        abstract=abstract,
        authors=["A. Author"],
        primary_category=fields.pop("primary_category", "cs.CL"),
        categories=fields.pop("categories", ["cs.CL"]),
        first_version_date=first or updated.date(),
        arxiv_updated_at=updated,
    )


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url) as connection:
        yield connection


def stored(connection, arxiv_id):
    return connection.execute(
        "SELECT version, abstract, first_version_date FROM paper WHERE arxiv_id = %s", (arxiv_id,)
    ).fetchone()


def categories(connection, arxiv_id):
    rows = connection.execute(
        "SELECT category FROM paper_category WHERE arxiv_id = %s ORDER BY 1", (arxiv_id,)
    )
    return [category for (category,) in rows]


# --- storing one Paper ---------------------------------------------------------------------


def test_new_paper_is_stored_with_every_category(connection):
    outcome = papers.store(connection, paper("1", categories=["cs.CV", "cs.CL"]))

    assert outcome == "new"
    assert categories(connection, "1") == ["cs.CL", "cs.CV"]


def test_same_version_again_changes_nothing(connection):
    papers.store(connection, paper("1"))

    assert papers.store(connection, paper("1")) == "unchanged"


def test_new_version_with_the_same_abstract_is_only_an_update(connection):
    papers.store(connection, paper("1"))

    assert papers.store(connection, paper("1", version=2)) == "updated"
    assert stored(connection, "1")[0] == 2


def test_new_version_with_another_abstract_is_reported_and_stored(connection):
    papers.store(connection, paper("1"))

    outcome = papers.store(connection, paper("1", version=2, abstract="Rewritten."))

    assert outcome == "abstract_changed"
    assert stored(connection, "1")[:2] == (2, "Rewritten.")


def test_date_of_the_first_version_never_changes(connection):
    first = date(2026, 1, 10)
    papers.store(connection, paper("1", first=first))

    papers.store(connection, paper("1", version=2, first=date(2026, 6, 1), abstract="New."))

    assert stored(connection, "1")[2] == first


def test_older_version_from_a_stale_page_does_not_overwrite(connection):
    papers.store(connection, paper("1", version=3, abstract="Third."))

    assert papers.store(connection, paper("1", version=2, abstract="Second.")) == "unchanged"
    assert stored(connection, "1")[:2] == (3, "Third.")


def test_older_version_from_a_stale_page_does_not_change_categories(connection):
    papers.store(connection, paper("1", version=2, categories=["cs.CL", "cs.AI"]))

    papers.store(connection, paper("1", version=1, categories=["cs.CL"]))

    assert categories(connection, "1") == ["cs.AI", "cs.CL"]


def test_category_added_without_a_new_version_is_picked_up(connection):
    papers.store(connection, paper("1", categories=["cs.LG"], primary_category="cs.LG"))

    papers.store(connection, paper("1", categories=["cs.LG", "cs.CL"], primary_category="cs.LG"))

    assert categories(connection, "1") == ["cs.CL", "cs.LG"]


def test_category_removed_from_the_paper_is_removed(connection):
    papers.store(connection, paper("1", categories=["cs.CL", "cs.AI"]))

    papers.store(connection, paper("1", version=2, categories=["cs.CL"]))

    assert categories(connection, "1") == ["cs.CL"]


# --- the daily job -------------------------------------------------------------------------


def test_daily_job_stores_new_and_revised_papers_and_counts_them(connection, fake_arxiv):
    papers.store(connection, paper("old", first=date(2026, 3, 1)))
    connection.commit()
    fake_arxiv.papers = [
        paper("new"),
        paper("old", version=2, first=date(2026, 3, 1), abstract="Rewritten."),
    ]

    counts = collect.collect_updates(connection, fake_arxiv, "cs.CL")

    assert dict(counts) == {"received": 2, "new": 1, "abstract_changed": 1}
    assert stored(connection, "old")[0] == 2


def test_daily_job_reads_back_to_its_previous_mark_plus_the_overlap(connection, fake_arxiv):
    mark = NOW - timedelta(days=3)
    connection.execute(
        "INSERT INTO collection_state (category, updated_through) VALUES ('cs.CL', %s)", (mark,)
    )
    inside = mark - collect.UPDATE_OVERLAP + timedelta(minutes=1)
    outside = mark - collect.UPDATE_OVERLAP - timedelta(minutes=1)
    fake_arxiv.papers = [
        paper("fresh"),
        paper("overlap", updated=inside),
        paper("seen", updated=outside),
    ]

    collect.collect_updates(connection, fake_arxiv, "cs.CL")

    assert stored(connection, "overlap") and not stored(connection, "seen")


def test_daily_job_moves_its_mark_to_the_newest_paper(connection, fake_arxiv):
    newest = NOW - timedelta(hours=2)
    fake_arxiv.papers = [paper("a", updated=newest), paper("b", updated=newest - timedelta(1))]

    collect.collect_updates(connection, fake_arxiv, "cs.CL")

    (mark,) = connection.execute("SELECT updated_through FROM collection_state").fetchone()
    assert mark == newest


def test_daily_job_keeps_its_mark_when_nothing_is_new(connection, fake_arxiv):
    mark = NOW - timedelta(days=1)
    connection.execute(
        "INSERT INTO collection_state (category, updated_through) VALUES ('cs.CL', %s)", (mark,)
    )

    assert collect.collect_updates(connection, fake_arxiv, "cs.CL") == {}

    assert connection.execute("SELECT updated_through FROM collection_state").fetchone() == (mark,)


def test_first_daily_run_of_a_slice_reads_a_week_back(connection, fake_arxiv):
    fake_arxiv.papers = [
        paper("recent", updated=NOW - timedelta(days=6)),
        paper("older", updated=NOW - timedelta(days=30)),
    ]

    collect.collect_updates(connection, fake_arxiv, "cs.CL")

    assert stored(connection, "recent") and not stored(connection, "older")


def test_revised_paper_from_before_the_history_is_not_stored(connection, fake_arxiv):
    fake_arxiv.papers = [paper("ancient", version=4, first=HISTORY_START - timedelta(days=1))]

    counts = collect.collect_updates(connection, fake_arxiv, "cs.CL")

    assert dict(counts) == {"received": 1, "before_history": 1}
    assert not stored(connection, "ancient")


def test_daily_job_refuses_an_implausible_flood_and_stores_nothing(
    connection, fake_arxiv, monkeypatch
):
    monkeypatch.setattr(collect, "MAX_PAPERS_PER_DAILY_RUN", 2)
    fake_arxiv.papers = [paper(str(n)) for n in range(4)]

    with pytest.raises(RuntimeError, match="refusing"):
        collect.collect_updates(connection, fake_arxiv, "cs.CL")
    connection.rollback()

    assert connection.execute("SELECT count(*) FROM paper").fetchone() == (0,)
    assert connection.execute("SELECT updated_through FROM collection_state").fetchone() == (None,)


def test_failed_daily_run_is_logged_and_leaves_the_mark_alone(database_url, fake_arxiv):
    def broken(category):
        raise RuntimeError("arXiv is down")

    fake_arxiv.latest_updated = broken

    with pytest.raises(RuntimeError):
        collect.run(database_url, fake_arxiv)

    with psycopg.connect(database_url) as connection:
        (run,) = latest_runs(connection)
        state = connection.execute("SELECT updated_through FROM collection_state").fetchone()
    assert (run.status, run.details["error"]) == ("failed", "RuntimeError: arXiv is down")
    assert state == (None,)


def test_daily_run_logs_what_it_did_per_slice(database_url, fake_arxiv):
    fake_arxiv.papers = [paper("new")]

    collect.run(database_url, fake_arxiv)

    with psycopg.connect(database_url) as connection:
        (run,) = latest_runs(connection)
    assert run.details == {"cs.CL": {"received": 1, "new": 1}, "requests": 1}


# --- the backfill --------------------------------------------------------------------------


def test_backfill_walks_every_day_from_the_start_of_the_history(connection, fake_arxiv):
    today = HISTORY_START + timedelta(days=5)
    fake_arxiv.papers = [paper("first", first=HISTORY_START), paper("too-recent", first=today)]

    counts = collect.backfill(connection, fake_arxiv, "cs.CL", today=today)

    last_day = today - timedelta(days=collect.BACKFILL_LAG_DAYS)
    assert fake_arxiv.days_asked[0] == HISTORY_START
    assert fake_arxiv.days_asked[-1] == last_day
    assert counts["days"] == len(fake_arxiv.days_asked) == 4
    assert stored(connection, "first") and not stored(connection, "too-recent")
    assert connection.execute("SELECT backfilled_through FROM collection_state").fetchone() == (
        last_day,
    )


def test_backfill_resumes_after_the_last_finished_day(connection, fake_arxiv):
    today = HISTORY_START + timedelta(days=10)
    collect.backfill(connection, fake_arxiv, "cs.CL", today=today - timedelta(days=4))
    fake_arxiv.days_asked.clear()

    collect.backfill(connection, fake_arxiv, "cs.CL", today=today)

    assert len(fake_arxiv.days_asked) == 4
    assert fake_arxiv.days_asked[0] == today - timedelta(days=4 + collect.BACKFILL_LAG_DAYS - 1)


def test_interrupted_backfill_keeps_the_days_it_finished(connection, fake_arxiv):
    fake_arxiv.papers = [paper("first", first=HISTORY_START)]
    served = fake_arxiv.submitted_on

    def fails_on_the_third_day(category, day):
        if day == HISTORY_START + timedelta(days=2):
            raise RuntimeError("arXiv is down")
        return served(category, day)

    fake_arxiv.submitted_on = fails_on_the_third_day

    with pytest.raises(RuntimeError):
        collect.backfill(connection, fake_arxiv, "cs.CL", today=HISTORY_START + timedelta(days=9))
    connection.rollback()

    assert stored(connection, "first")
    assert connection.execute("SELECT backfilled_through FROM collection_state").fetchone() == (
        HISTORY_START + timedelta(days=1),
    )


# --- one collection job at a time -------------------------------------------------------------


def test_second_collection_job_is_skipped_not_failed_while_one_is_running(database_url, fake_arxiv):
    fake_arxiv.papers = [paper("new")]
    with psycopg.connect(database_url) as running:
        running.execute("SELECT pg_advisory_lock(%s)", (collect.COLLECTION_LOCK,))

        collect.run(database_url, fake_arxiv)

    with psycopg.connect(database_url) as connection:
        (run,) = latest_runs(connection)
        assert (run.status, run.details) == (
            "skipped",
            {"skipped": "another collection job is running"},
        )
        assert not stored(connection, "new")
        assert fake_arxiv.requests == 0


def test_collection_runs_again_once_the_other_job_is_done(database_url, fake_arxiv):
    with psycopg.connect(database_url) as finished:
        finished.execute("SELECT pg_advisory_lock(%s)", (collect.COLLECTION_LOCK,))

    collect.run(database_url, fake_arxiv)
    collect.run_backfill(database_url, fake_arxiv)


def test_limit_of_the_daily_job_can_be_raised_for_one_run(connection, fake_arxiv, monkeypatch):
    monkeypatch.setattr(collect, "MAX_PAPERS_PER_DAILY_RUN", 2)
    fake_arxiv.papers = [paper(str(n)) for n in range(4)]

    with pytest.raises(RuntimeError, match="--max-papers"):
        collect.collect_updates(connection, fake_arxiv, "cs.CL")
    connection.rollback()
    counts = collect.collect_updates(connection, fake_arxiv, "cs.CL", max_papers=10)

    assert counts["new"] == 4


def test_limit_of_zero_is_not_read_as_no_limit_given(connection, fake_arxiv):
    fake_arxiv.papers = [paper("only")]

    with pytest.raises(RuntimeError, match="more than 0 papers"):
        collect.collect_updates(connection, fake_arxiv, "cs.CL", max_papers=0)


def test_jobs_refuse_a_pooled_connection(database_url, fake_arxiv, monkeypatch):
    class Pooled:
        host = "ep-example-pooler.eu-central-1.aws.neon.tech"

    monkeypatch.setattr(psycopg.Connection, "info", Pooled(), raising=False)

    with pytest.raises(RuntimeError, match="direct URL"):
        collect.run(database_url, fake_arxiv)
    assert fake_arxiv.requests == 0
