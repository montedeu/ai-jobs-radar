from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from research_radar.extraction import start_run
from research_radar.runs import current_run, missing_papers, promote
from research_radar.settings import EXTRACTION_MODEL

T0 = datetime(2025, 12, 1, 12, tzinfo=UTC)
RUN_AT = T0 + timedelta(hours=1)
LATER = T0 + timedelta(hours=2)


@pytest.fixture
def connection(database_url):
    # Autocommit, like the callers of extraction: a test's own read leaves no transaction open.
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(
    connection,
    arxiv_id,
    *,
    categories=("cs.CL",),
    first_version_date="2025-11-03",
    collected_at=T0,
    abstract_hash="h",
):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at, collected_at)
        VALUES (%s, 1, 't', 'a', %s, '[]', %s, %s, '2025-11-03', %s)
        """,
        (arxiv_id, abstract_hash, categories[0], first_version_date, collected_at),
    )
    for category in categories:
        connection.execute(
            "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, %s)", (arxiv_id, category)
        )


def new_run(connection, *, full=True, created_at=RUN_AT):
    run_id = start_run(connection, model=EXTRACTION_MODEL, prompt_hash="stub", full=full)
    connection.execute(
        "UPDATE extraction_run SET created_at = %s WHERE id = %s", (created_at, run_id)
    )
    return run_id


def mark_done(connection, run_id, *arxiv_ids, abstract_hash="h"):
    for arxiv_id in arxiv_ids:
        connection.execute(
            "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
            " output_tokens) VALUES (%s, %s, %s, 1, 1)",
            (run_id, arxiv_id, abstract_hash),
        )


CL = ["2511.00001", "2511.00002", "2511.00003"]


@pytest.fixture
def papers(connection):
    for arxiv_id in CL:
        add_paper(connection, arxiv_id)
    add_paper(connection, "2511.00004", categories=("cs.CV",))
    add_paper(connection, "2511.00005", collected_at=LATER)


def test_no_current_run_before_any_promotion(connection, papers):
    assert current_run(connection) is None
    new_run(connection)
    assert current_run(connection) is None


def test_start_run_records_whether_the_run_is_full(connection):
    full = new_run(connection)
    pilot = new_run(connection, full=False)
    rows = connection.execute('SELECT id, "full" FROM extraction_run ORDER BY id').fetchall()
    assert rows == [(full, True), (pilot, False)]
    assert start_run(connection, model=EXTRACTION_MODEL, prompt_hash="x") > pilot
    assert connection.execute('SELECT "full" FROM extraction_run ORDER BY id DESC').fetchone() == (
        False,
    )


def test_full_run_covering_its_scope_becomes_current(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run
    assert current_run(connection) == run
    (since,) = connection.execute(
        "SELECT current_since FROM extraction_run WHERE id = %s", (run,)
    ).fetchone()
    assert since is not None


def test_full_run_missing_a_paper_does_not_become_current(connection, papers):
    first = new_run(connection)
    mark_done(connection, first, *CL)
    assert promote(connection) == first
    second = new_run(connection, created_at=RUN_AT + timedelta(minutes=1))
    mark_done(connection, second, *CL[:2])
    assert missing_papers(connection, second) == ["2511.00003"]
    assert promote(connection) is None


def test_pilot_run_covering_everything_never_becomes_current(connection, papers):
    pilot = new_run(connection, full=False)
    mark_done(connection, pilot, *CL)
    assert missing_papers(connection, pilot) == []
    assert promote(connection) is None
    assert current_run(connection) is None


def test_papers_collected_after_the_run_started_are_outside_its_scope(connection, papers):
    run = new_run(connection)
    assert "2511.00005" not in missing_papers(connection, run)
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == []
    # A paper collected at the very moment the run was created is inside the scope.
    add_paper(connection, "2511.00006", collected_at=RUN_AT)
    assert missing_papers(connection, run) == ["2511.00006"]


def test_previous_run_stays_current_until_the_new_one_covers_its_scope(connection, papers):
    first = new_run(connection)
    mark_done(connection, first, *CL)
    assert promote(connection) == first
    second = new_run(connection, created_at=RUN_AT + timedelta(minutes=1))
    mark_done(connection, second, *CL[:2])
    assert promote(connection) is None
    assert current_run(connection) == first
    mark_done(connection, second, CL[2])
    assert promote(connection) == second
    assert current_run(connection) == second


def test_older_full_run_does_not_replace_a_newer_current_one(connection, papers):
    older = new_run(connection)
    newer = new_run(connection, created_at=RUN_AT + timedelta(minutes=1))
    mark_done(connection, newer, *CL)
    assert promote(connection) == newer
    mark_done(connection, older, *CL)
    assert promote(connection) is None
    assert current_run(connection) == newer


def test_of_two_covering_full_runs_the_newer_becomes_current(connection, papers):
    older = new_run(connection)
    newer = new_run(connection, created_at=RUN_AT + timedelta(minutes=1))
    mark_done(connection, older, *CL)
    mark_done(connection, newer, *CL)
    assert promote(connection) == newer
    assert current_run(connection) == newer


def test_newer_run_that_is_unfinished_does_not_block_an_older_finished_one(connection, papers):
    older = new_run(connection)
    newer = new_run(connection, created_at=RUN_AT + timedelta(minutes=1))
    mark_done(connection, older, *CL)
    mark_done(connection, newer, CL[0])
    assert promote(connection) == older
    assert current_run(connection) == older


def test_papers_of_other_slices_and_before_history_start_are_outside_the_scope(connection, papers):
    add_paper(connection, "2511.00007", first_version_date="2025-10-31")
    # Cross-listed: cs.CL is one of its categories, though not the primary one.
    add_paper(connection, "2511.00008", categories=("cs.CV", "cs.CL"))
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == ["2511.00008"]


def test_a_paper_whose_abstract_changed_after_the_mark_is_still_covered(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL)
    connection.execute("UPDATE paper SET abstract_hash = 'new' WHERE arxiv_id = '2511.00001'")
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_full_run_over_an_empty_scope_becomes_current(connection):
    run = new_run(connection)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_promote_is_idempotent(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert promote(connection) == run
    (since,) = connection.execute(
        "SELECT current_since FROM extraction_run WHERE id = %s", (run,)
    ).fetchone()
    assert promote(connection) is None
    assert current_run(connection) == run
    assert connection.execute(
        "SELECT current_since FROM extraction_run WHERE id = %s", (run,)
    ).fetchone() == (since,)


def test_missing_papers_are_sorted_by_arxiv_id(connection, papers):
    run = new_run(connection)
    assert missing_papers(connection, run) == sorted(CL)


def test_missing_papers_of_an_unknown_run_is_an_error(connection, papers):
    with pytest.raises(LookupError):
        missing_papers(connection, 999999)
