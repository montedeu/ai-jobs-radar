import psycopg
import pytest
from fakes import FakeModel

from research_radar import cli, collect, daily_extraction, extraction, runs
from research_radar.daily_extraction import TooManyPapers, extract_daily
from research_radar.extraction import ExtractionResult, RunConfigMismatch, start_run
from research_radar.mentions import PaperMentions
from research_radar.papers import abstract_hash
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS, EXTRACTION_MODEL


@pytest.fixture
def connection(database_url):
    # Autocommit: the extraction steps refuse a connection with a transaction open.
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(connection, arxiv_id, abstract="an abstract", *, categories=("cs.CL",)):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%s, 1, 't', %s, %s, '[]', %s, '2025-11-03', '2025-11-03')
        """,
        (arxiv_id, abstract, abstract_hash(abstract), categories[0]),
    )
    for category in categories:
        connection.execute(
            "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, %s)", (arxiv_id, category)
        )


def mark_done(connection, run_id, arxiv_id, abstract="an abstract"):
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, %s, %s, 1, 1)",
        (run_id, arxiv_id, abstract_hash(abstract)),
    )


def fail_attempts(connection, run_id, arxiv_id, abstract="an abstract"):
    for _ in range(EXTRACTION_MAX_ATTEMPTS):
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'collected', 0) RETURNING id",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash, outcome)"
            " VALUES (%s, %s, %s, 'invalid')",
            (batch_id, arxiv_id, abstract_hash(abstract)),
        )


def make_current(connection, **done):
    """A full run that has covered its scope and been switched to; ``done`` maps id -> abstract
    the paper was extracted with."""
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    for arxiv_id, abstract in done.items():
        mark_done(connection, run_id, arxiv_id, abstract)
    assert runs.promote(connection) == run_id
    return run_id


def answer(arxiv_id):
    return ExtractionResult(arxiv_id, PaperMentions(mentions=[]), 10, 5)


def factory(model):
    calls = []

    def make():
        calls.append(True)
        return model

    make.calls = calls
    return make


@pytest.fixture
def current(connection):
    """Run 1, full and current. Of its papers one is done with the same abstract, one with
    another, and one (added after the run) is new."""
    add_paper(connection, "2511.00001", "same")
    add_paper(connection, "2511.00002", "now")
    run_id = make_current(connection, **{"2511.00001": "same", "2511.00002": "before"})
    add_paper(connection, "2511.00003", "new")
    return run_id


def job_rows(connection):
    return connection.execute("SELECT job, status, details FROM job_run ORDER BY id").fetchall()


def test_without_current_run_nothing_is_sent_and_no_client_is_created(connection):
    add_paper(connection, "2511.00001")
    make = factory(FakeModel())

    details = extract_daily(connection, make)

    assert make.calls == []
    assert (details["current_run"], details["due"], details["submitted_batch"]) == (None, 0, None)


def test_new_paper_is_sent_with_the_current_run(connection, current):
    model = FakeModel()

    details = extract_daily(connection, factory(model))

    assert sorted(r.arxiv_id for r in model.sent[0]) == ["2511.00002", "2511.00003"]
    assert details["due"] == 2
    batch_run = connection.execute(
        "SELECT run_id FROM extraction_batch WHERE id = %s", (details["submitted_batch"],)
    ).fetchone()[0]
    assert batch_run == current


def test_paper_with_completion_mark_and_same_hash_is_not_sent(connection, current):
    model = FakeModel()

    extract_daily(connection, factory(model))

    assert "2511.00001" not in [r.arxiv_id for r in model.sent[0]]


def test_paper_with_changed_abstract_is_sent_again(connection, current):
    model = FakeModel()

    extract_daily(connection, factory(model))

    assert "2511.00002" in [r.arxiv_id for r in model.sent[0]]


def test_paper_in_flight_is_not_sent_again(connection, current):
    model = FakeModel(ended=False)
    extract_daily(connection, factory(model))

    details = extract_daily(connection, factory(model))

    assert len(model.sent) == 1
    assert details["due"] == 0
    assert details["batches_collected"] == 1
    assert details["collected"] == {"pending": 2}


def test_paper_in_a_sending_batch_is_not_sent_again(connection, current):
    batch_id = connection.execute(
        "INSERT INTO extraction_batch (run_id, estimated_cost_usd) VALUES (%s, 0) RETURNING id",
        (current,),
    ).fetchone()[0]
    for arxiv_id, abstract in (("2511.00002", "now"), ("2511.00003", "new")):
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash)"
            " VALUES (%s, %s, %s)",
            (batch_id, arxiv_id, abstract_hash(abstract)),
        )
    make = factory(FakeModel())

    details = extract_daily(connection, make)

    assert make.calls == []
    assert details["due"] == 0
    assert details["collected"] == {"unsent": 2}


def test_batch_sent_yesterday_is_collected_today(connection, current):
    model = FakeModel(answer("2511.00002"), answer("2511.00003"), ended=False)
    extract_daily(connection, factory(model))
    model.ended = True

    details = extract_daily(connection, factory(model))

    assert details["collected"]["extracted"] == 2
    assert details["due"] == 0
    done = connection.execute(
        'SELECT arxiv_id FROM paper_extraction WHERE run_id = %s ORDER BY arxiv_id COLLATE "C"',
        (current,),
    ).fetchall()
    assert [row[0] for row in done] == ["2511.00001", "2511.00002", "2511.00003"]


def test_full_run_batches_are_collected_and_the_run_is_promoted(connection, current):
    add_paper(connection, "2511.00004", "full")
    full = start_run(connection, model=EXTRACTION_MODEL, full=True)
    model = FakeModel(*[answer(f"2511.0000{n}") for n in range(1, 5)])
    batch_id = extraction.submit(connection, model, full, ["2511.00001", "2511.00002"])
    # The rest of the scope was done another way; only this batch is still out.
    mark_done(connection, full, "2511.00003", "new")
    mark_done(connection, full, "2511.00004", "full")
    assert runs.current_run(connection) == current

    details = extract_daily(connection, factory(model))

    assert details["batches_collected"] == 1
    assert details["promoted"] == full
    assert details["current_run"] == full
    assert runs.current_run(connection) == full
    assert connection.execute(
        "SELECT status FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone() == ("collected",)


def test_over_the_limit_fails_and_sends_nothing(database_url, connection, current):
    model = FakeModel()

    with pytest.raises(TooManyPapers):
        daily_extraction.run(database_url, factory(model), max_papers=1)

    assert model.calls == []
    (row,) = [r for r in job_rows(connection) if r[0] == "extract"]
    assert row[1] == "failed"
    assert "TooManyPapers" in row[2]["error"]


def test_paper_the_run_gave_up_on_is_not_sent_again(connection, current):
    fail_attempts(connection, current, "2511.00003", "new")
    model = FakeModel()

    details = extract_daily(connection, factory(model))

    assert [r.arxiv_id for r in model.sent[0]] == ["2511.00002"]
    assert details["gave_up"] == 1


def test_given_up_papers_do_not_count_against_the_limit_or_create_a_client(connection, current):
    connection.execute(
        "UPDATE paper_extraction SET abstract_hash = %s WHERE arxiv_id = '2511.00002'",
        (abstract_hash("now"),),
    )
    fail_attempts(connection, current, "2511.00003", "new")
    make = factory(FakeModel())

    details = extract_daily(connection, make, max_papers=0)

    assert make.calls == []
    assert (details["due"], details["gave_up"]) == (0, 1)


def test_job_log_has_submitted_collected_and_spent(database_url, connection, current):
    model = FakeModel()

    daily_extraction.run(database_url, factory(model))

    (row,) = [r for r in job_rows(connection) if r[0] == "extract"]
    assert row[1] == "succeeded"
    details = row[2]
    assert details["submitted"] == 2
    assert details["collected"] == {}
    assert details["spent_usd"] == str(extraction.spent_usd(connection))
    assert float(details["spent_usd"]) > 0


def test_config_mismatch_fails_the_step_but_keeps_collected_and_promoted(
    database_url, connection, current
):
    add_paper(connection, "2511.00004", "full")
    full = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, full, "2511.00001", "same")
    mark_done(connection, full, "2511.00002", "now")
    mark_done(connection, full, "2511.00003", "new")
    mark_done(connection, full, "2511.00004", "full")
    connection.execute("UPDATE extraction_run SET prompt_hash = 'other' WHERE id = %s", (full,))
    add_paper(connection, "2511.00005", "later")

    with pytest.raises(RunConfigMismatch):
        daily_extraction.run(database_url, factory(FakeModel()))

    (row,) = [r for r in job_rows(connection) if r[0] == "extract"]
    assert row[1] == "failed"
    assert row[2]["promoted"] == full
    assert row[2]["collected"] == {}
    assert "RunConfigMismatch" in row[2]["error"]


def test_budget_exceeded_fails_the_step_and_keeps_what_was_collected(
    database_url, connection, current, monkeypatch
):
    def refuse(*args, **kwargs):
        raise extraction.BudgetExceeded("over the ceiling")

    monkeypatch.setattr(extraction, "submit", refuse)

    with pytest.raises(extraction.BudgetExceeded):
        daily_extraction.run(database_url, factory(FakeModel()))

    (row,) = [r for r in job_rows(connection) if r[0] == "extract"]
    assert row[1] == "failed"
    assert row[2]["due"] == 2
    assert "over the ceiling" in row[2]["error"]


def test_empty_database_succeeds_and_sends_nothing(database_url, connection):
    make = factory(FakeModel())

    daily_extraction.run(database_url, make)

    assert make.calls == []
    assert [(r[0], r[1]) for r in job_rows(connection)] == [("extract", "succeeded")]


def test_task_never_starts_a_run(connection, current):
    before = connection.execute("SELECT count(*) FROM extraction_run").fetchone()[0]

    extract_daily(connection, factory(FakeModel()))

    assert connection.execute("SELECT count(*) FROM extraction_run").fetchone()[0] == before


class BrokenArxiv:
    requests = 0

    def latest_updated(self, category):
        raise RuntimeError("arXiv is down")


def test_daily_command_runs_extraction_after_a_failed_collection(
    database_url, connection, current, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(collect, "ArxivClient", BrokenArxiv)
    model = FakeModel()
    monkeypatch.setattr(extraction, "AnthropicModelClient", lambda: model)

    with pytest.raises(SystemExit) as stopped:
        cli.main(["daily"])

    assert stopped.value.code not in (0, None)
    assert [(r[0], r[1]) for r in job_rows(connection)] == [
        ("collect", "failed"),
        ("extract", "succeeded"),
    ]
    assert len(model.sent) == 1


def test_daily_command_runs_collection_when_extraction_fails(
    database_url, connection, current, fake_arxiv, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(collect, "ArxivClient", lambda: fake_arxiv)

    def no_key():
        raise RuntimeError("no API key")

    monkeypatch.setattr(extraction, "AnthropicModelClient", no_key)

    with pytest.raises(SystemExit) as stopped:
        cli.main(["daily"])

    assert stopped.value.code not in (0, None)
    assert [(r[0], r[1]) for r in job_rows(connection)] == [
        ("collect", "succeeded"),
        ("extract", "failed"),
    ]
