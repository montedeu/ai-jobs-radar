import psycopg
import pytest

from research_radar.extraction import start_run
from research_radar.runs import current_run, head_candidate, missing_papers, promote, run_state
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS, EXTRACTION_MODEL


@pytest.fixture
def connection(database_url):
    # Autocommit, like the callers of extraction: no transaction is left open by a read.
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(connection, arxiv_id, *, categories=("cs.CL",), first_version_date="2025-11-03"):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%s, 1, 't', 'a', 'h', '[]', %s, %s, '2025-11-03')
        """,
        (arxiv_id, categories[0], first_version_date),
    )
    for category in categories:
        connection.execute(
            "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, %s)", (arxiv_id, category)
        )


def new_run(connection, *, full=True):
    return start_run(connection, model=EXTRACTION_MODEL, full=full)


def mark_done(connection, run_id, *arxiv_ids):
    for arxiv_id in arxiv_ids:
        connection.execute(
            "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
            " output_tokens) VALUES (%s, %s, 'h', 1, 1)",
            (run_id, arxiv_id),
        )


def fail_attempts(connection, run_id, arxiv_id, count, *, abstract_hash="h", outcome="invalid"):
    """Collected batches that carried the paper and ended with the given outcome for it."""
    for _ in range(count):
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'collected', 0) RETURNING id",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash, outcome)"
            " VALUES (%s, %s, %s, %s)",
            (batch_id, arxiv_id, abstract_hash, outcome),
        )


def add_filler(connection, count):
    """Papers that make the scope big enough for a given-up paper to be under the share."""
    ids = [f"2512.{n:05d}" for n in range(count)]
    for arxiv_id in ids:
        add_paper(connection, arxiv_id)
    return ids


CL = ["2511.00001", "2511.00002", "2511.00003"]


@pytest.fixture
def papers(connection):
    for arxiv_id in CL:
        add_paper(connection, arxiv_id)
    add_paper(connection, "2511.00004", categories=("cs.CV",))


def test_no_current_run_before_any_promotion(connection, papers):
    assert current_run(connection) is None
    new_run(connection)
    assert current_run(connection) is None


def test_start_run_records_whether_the_run_is_full(connection):
    full = new_run(connection)
    pilot = new_run(connection, full=False)
    default = start_run(connection, model=EXTRACTION_MODEL)
    rows = connection.execute("SELECT id, is_full FROM extraction_run ORDER BY id").fetchall()
    assert rows == [(full, True), (pilot, False), (default, False)]


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
    run = new_run(connection)
    mark_done(connection, run, *CL[:2])
    assert missing_papers(connection, run) == ["2511.00003"]
    assert promote(connection) is None
    assert current_run(connection) is None


def test_pilot_run_covering_everything_never_becomes_current(connection, papers):
    pilot = new_run(connection, full=False)
    mark_done(connection, pilot, *CL)
    assert promote(connection) is None
    assert current_run(connection) is None


def test_papers_collected_after_the_run_started_are_outside_its_scope(connection, papers):
    run = new_run(connection)
    add_paper(connection, "2511.00005")
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_paper_that_gained_the_category_after_the_start_is_outside_the_scope(connection, papers):
    run = new_run(connection)
    connection.execute("INSERT INTO paper_category VALUES ('2511.00004', 'cs.CL')")
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_papers_of_other_slices_and_before_history_start_are_outside_the_scope(connection, papers):
    add_paper(connection, "2511.00007", first_version_date="2025-10-31")
    add_paper(connection, "2511.00008", categories=("cs.CV", "cs.CL"))  # cross-listed
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == ["2511.00008"]


def test_papers_with_an_id_that_cannot_be_extracted_are_outside_the_scope(connection, papers):
    add_paper(connection, "hep-th/9901001")
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_previous_run_stays_current_until_the_new_one_covers_its_scope(connection, papers):
    first = new_run(connection)
    mark_done(connection, first, *CL)
    assert promote(connection) == first
    second = new_run(connection)
    mark_done(connection, second, *CL[:2])
    assert promote(connection) is None
    assert current_run(connection) == first
    mark_done(connection, second, CL[2])
    assert promote(connection) == second
    assert current_run(connection) == second


def test_older_full_run_does_not_replace_a_newer_current_one(connection, papers):
    older = new_run(connection)
    newer = new_run(connection)
    mark_done(connection, newer, *CL)
    assert promote(connection) == newer
    mark_done(connection, older, *CL)
    assert promote(connection) is None
    assert current_run(connection) == newer


def test_of_two_covering_full_runs_the_newer_becomes_current(connection, papers):
    older = new_run(connection)
    newer = new_run(connection)
    mark_done(connection, older, *CL)
    mark_done(connection, newer, *CL)
    assert promote(connection) == newer
    assert current_run(connection) == newer


def test_newer_run_that_is_unfinished_does_not_block_an_older_finished_one(connection, papers):
    older = new_run(connection)
    newer = new_run(connection)
    mark_done(connection, older, *CL)
    mark_done(connection, newer, CL[0])
    assert promote(connection) == older
    assert current_run(connection) == older


def test_a_paper_whose_abstract_changed_after_the_mark_is_still_covered(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL)
    connection.execute("UPDATE paper SET abstract_hash = 'new' WHERE arxiv_id = '2511.00001'")
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_paper_given_up_on_does_not_block_the_switch(connection, papers):
    filler = add_filler(connection, 97)  # 100 papers in the scope: one given up is 1%
    run = new_run(connection)
    mark_done(connection, run, *CL[:2], *filler)
    fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_paper_with_fewer_invalid_answers_still_blocks_the_switch(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL[:2])
    fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS - 1)
    assert missing_papers(connection, run) == [CL[2]]
    assert promote(connection) is None


def test_batches_the_api_failed_are_not_attempts(connection, papers):
    filler = add_filler(connection, 97)
    run = new_run(connection)
    mark_done(connection, run, *CL[:2], *filler)
    for outcome in ("expired", "canceled", "stale", "extracted", None):
        fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS, outcome=outcome)
    assert missing_papers(connection, run) == [CL[2]]
    assert promote(connection) is None


def test_a_request_the_api_always_rejects_is_given_up_on(connection, papers):
    filler = add_filler(connection, 97)
    run = new_run(connection)
    mark_done(connection, run, *CL[:2], *filler)
    fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS, outcome="errored")
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_one_given_up_paper_never_locks_a_small_scope(connection, papers):
    filler = add_filler(connection, 57)  # 60 papers: one is 1.7%, over the share
    run = new_run(connection)
    mark_done(connection, run, *CL[:2], *filler)
    fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS)
    assert promote(connection) == run


def test_two_given_up_papers_of_a_small_scope_do_lock_it(connection, papers):
    filler = add_filler(connection, 57)
    run = new_run(connection)
    mark_done(connection, run, CL[0], *filler)
    for arxiv_id in CL[1:]:
        fail_attempts(connection, run, arxiv_id, EXTRACTION_MAX_ATTEMPTS)
    assert promote(connection) is None


def test_invalid_answers_on_an_older_abstract_do_not_count(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL[:2])
    fail_attempts(connection, run, CL[2], EXTRACTION_MAX_ATTEMPTS, abstract_hash="old")
    assert missing_papers(connection, run) == [CL[2]]


def test_run_with_more_than_one_percent_given_up_does_not_become_current(connection, papers):
    filler = add_filler(connection, 97)
    run = new_run(connection)
    mark_done(connection, run, CL[0], *filler)
    for arxiv_id in CL[1:]:
        fail_attempts(connection, run, arxiv_id, EXTRACTION_MAX_ATTEMPTS)
    assert missing_papers(connection, run) == []
    assert promote(connection) is None
    assert current_run(connection) is None


def test_full_run_over_an_empty_scope_becomes_current(connection):
    run = new_run(connection)
    assert missing_papers(connection, run) == []
    assert promote(connection) == run


def test_promote_is_idempotent(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, *CL)
    assert promote(connection) == run
    query = "SELECT current_since FROM extraction_run WHERE id = %s"
    (since,) = connection.execute(query, (run,)).fetchone()
    assert promote(connection) is None
    assert current_run(connection) == run
    assert connection.execute(query, (run,)).fetchone() == (since,)


def test_promote_refuses_a_connection_with_a_transaction_open(connection, papers):
    new_run(connection)
    with connection.transaction(), pytest.raises(ValueError, match="transaction"):
        promote(connection)


def test_missing_papers_are_sorted_by_arxiv_id(connection, papers):
    run = new_run(connection)
    assert missing_papers(connection, run) == sorted(CL)


def test_missing_papers_of_a_pilot_run_is_an_error(connection, papers):
    pilot = new_run(connection, full=False)
    with pytest.raises(ValueError, match="full"):
        missing_papers(connection, pilot)


def test_start_run_refuses_a_connection_with_a_transaction_open(connection):
    with connection.transaction(), pytest.raises(ValueError, match="transaction"):
        new_run(connection)


def test_missing_papers_of_an_unknown_run_is_an_error(connection, papers):
    with pytest.raises(LookupError):
        missing_papers(connection, 999999)


def test_run_state_counts_what_holds_the_run_back_and_what_it_gave_up_on(connection, papers):
    run = new_run(connection)
    mark_done(connection, run, CL[0])
    fail_attempts(connection, run, CL[1], EXTRACTION_MAX_ATTEMPTS)

    state = run_state(connection, run)

    assert (state.missing, state.given_up, state.scope) == (1, 1, 3)
    assert state.blocked is False


def test_run_state_is_blocked_at_the_same_limit_as_promote(connection, papers):
    run = new_run(connection)
    for arxiv_id in CL[:2]:
        fail_attempts(connection, run, arxiv_id, EXTRACTION_MAX_ATTEMPTS)

    # Papers are still missing: a blocked run is blocked whatever else is left.
    state = run_state(connection, run)

    assert (state.missing, state.given_up, state.blocked) == (1, 2, True)


def test_head_candidate_is_the_newest_unfinished_full_run_after_the_current(connection, papers):
    assert head_candidate(connection) is None
    older = new_run(connection)
    pilot = new_run(connection, full=False)
    newer = new_run(connection)
    assert head_candidate(connection) == newer
    mark_done(connection, older, *CL)
    assert promote(connection) == older
    assert head_candidate(connection) == newer
    mark_done(connection, newer, *CL)
    assert promote(connection) == newer
    assert head_candidate(connection) is None
    assert pilot not in (older, newer)
