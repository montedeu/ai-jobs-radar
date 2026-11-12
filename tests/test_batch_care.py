"""Taking care of batches: giving up a stuck one, and keeping error texts inside."""

import psycopg
import pytest
from fakes import FakeModel
from psycopg.types.json import Jsonb
from starlette.testclient import TestClient

from research_radar import cli, daily_extraction, extraction, full_run, mcp_server, runs
from research_radar.extraction import start_run
from research_radar.papers import abstract_hash
from research_radar.settings import EXTRACTION_MODEL

TOKEN = "test-token"


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def connection(database_url):
    # Autocommit: the extraction steps refuse a connection with a transaction open.
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(connection, arxiv_id, abstract="an abstract"):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%s, 1, 't', %s, %s, '[]', 'cs.CL', '2025-11-03', '2025-11-03')
        """,
        (arxiv_id, abstract, abstract_hash(abstract)),
    )
    connection.execute(
        "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, 'cs.CL')", (arxiv_id,)
    )


def mark_done(connection, run_id, arxiv_id):
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, %s, %s, 1, 1)",
        (run_id, arxiv_id, abstract_hash("an abstract")),
    )


def add_batch(connection, run_id, status, arxiv_ids, estimate="1.5"):
    """A batch row as the run left it, with its papers; nothing is sent anywhere."""
    batch_id = connection.execute(
        "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd, provider_batch_id)"
        " VALUES (%s, %s, %s, %s) RETURNING id",
        (run_id, status, estimate, None if status == "sending" else f"provider-{status}"),
    ).fetchone()[0]
    for arxiv_id in arxiv_ids:
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash)"
            " VALUES (%s, %s, %s)",
            (batch_id, arxiv_id, abstract_hash("an abstract")),
        )
    return batch_id


def status_of(connection, batch_id):
    return connection.execute(
        "SELECT status FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone()[0]


@pytest.fixture
def current(connection):
    """Run 1 is full and current; its paper is done. Two papers were added after it."""
    add_paper(connection, "2511.00001")
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, "2511.00001")
    assert runs.promote(connection) == run_id
    add_paper(connection, "2511.00002")
    add_paper(connection, "2511.00003")
    return run_id


def test_abandoning_a_sending_batch_of_the_current_run_frees_its_papers_for_the_daily_job(
    connection, current
):
    batch_id = add_batch(connection, current, "sending", ["2511.00002", "2511.00003"])
    assert daily_extraction._due(connection, current) == ([], 0)

    papers = extraction.abandon_batch(connection, batch_id)

    assert papers == 2
    assert status_of(connection, batch_id) == "abandoned"
    assert daily_extraction._due(connection, current) == (["2511.00002", "2511.00003"], 0)
    model = FakeModel()
    details = daily_extraction.extract_daily(connection, lambda: model)
    assert details["submitted"] == 2
    assert sorted(r.arxiv_id for r in model.sent[0]) == ["2511.00002", "2511.00003"]


def test_abandoning_a_submitted_batch_keeps_its_estimate_in_spent(connection, current):
    batch_id = add_batch(connection, current, "submitted", ["2511.00002"], estimate="2.25")
    before = extraction.spent_usd(connection)

    assert extraction.abandon_batch(connection, batch_id) == 1

    assert status_of(connection, batch_id) == "abandoned"
    assert extraction.spent_usd(connection) == before
    assert before == 2.25
    # collect passes it by without touching the network
    model = FakeModel()
    assert extraction.collect(connection, model, batch_id) == {"abandoned": 1}
    assert model.calls == []


@pytest.mark.parametrize("status", ["collected", "rejected", "abandoned"])
def test_collected_or_rejected_batch_cannot_be_abandoned(connection, current, status):
    batch_id = add_batch(connection, current, status, ["2511.00002"])

    with pytest.raises(ValueError, match=status):
        extraction.abandon_batch(connection, batch_id)

    assert status_of(connection, batch_id) == status


def test_an_unknown_batch_is_a_lookup_error(connection, current):
    with pytest.raises(LookupError):
        extraction.abandon_batch(connection, 999)


def test_abandon_batch_needs_a_connection_with_no_transaction_open(connection, current):
    batch_id = add_batch(connection, current, "sending", ["2511.00002"])
    connection.autocommit = False
    connection.execute("SELECT 1")

    with pytest.raises(ValueError, match="transaction"):
        extraction.abandon_batch(connection, batch_id)

    connection.rollback()
    assert status_of(connection, batch_id) == "sending"


@pytest.fixture
def cli_database(monkeypatch, database_url):
    monkeypatch.setenv("DATABASE_URL", database_url)


def test_abandon_batch_without_confirm_changes_nothing(connection, current, cli_database, capsys):
    batch_id = add_batch(connection, current, "sending", ["2511.00002", "2511.00003"])
    before = extraction.spent_usd(connection)

    cli.main(["abandon-batch", str(batch_id)])

    shown = capsys.readouterr().out
    assert status_of(connection, batch_id) == "sending"
    assert extraction.spent_usd(connection) == before
    assert f"run {current}" in shown
    assert "sending" in shown
    assert "2 papers" in shown
    assert "$1.5" in shown
    assert "--confirm" in shown


def test_abandon_batch_with_confirm_gives_the_batch_up_and_warns_about_paying_twice(
    connection, current, cli_database, capsys
):
    batch_id = add_batch(connection, current, "submitted", ["2511.00002"])

    cli.main(["abandon-batch", str(batch_id), "--confirm"])

    shown = capsys.readouterr().out
    assert status_of(connection, batch_id) == "abandoned"
    assert "paid for again" in shown
    assert "$1.5" in shown


def test_abandon_batch_of_an_unknown_id_exits_with_an_error(connection, current, cli_database):
    with pytest.raises(SystemExit) as stop:
        cli.main(["abandon-batch", "999", "--confirm"])

    assert stop.value.code not in (0, None)
    assert "999" in str(stop.value.code)


def test_abandon_batch_of_a_collected_batch_exits_with_an_error(connection, current, cli_database):
    batch_id = add_batch(connection, current, "collected", ["2511.00002"])

    with pytest.raises(SystemExit) as stop:
        cli.main(["abandon-batch", str(batch_id), "--confirm"])

    assert stop.value.code not in (0, None)
    assert status_of(connection, batch_id) == "collected"


class BrokenBatches(FakeModel):
    """Two batches fail to be collected: one with a database error, one with a plain one."""

    def batch_results(self, batch_id):
        if batch_id == "batch-1":
            raise psycopg.errors.InvalidPassword("host=secret-host user=secret-user")
        if batch_id == "batch-2":
            raise RuntimeError("results of secret-paper expired")
        return super().batch_results(batch_id)


def test_batch_errors_keep_only_the_error_type(connection, current, database_url):
    model = BrokenBatches()
    first = extraction.submit(connection, model, current, ["2511.00002"])
    second = extraction.submit(connection, model, current, ["2511.00003"])

    with pytest.raises(RuntimeError, match="not collected"):
        daily_extraction.run(database_url, lambda: model)

    details = connection.execute("SELECT details FROM job_run WHERE job = 'extract'").fetchone()[0]
    assert details["batch_errors"] == {
        str(first): "InvalidPassword (SQLSTATE 28P01)",
        str(second): "RuntimeError",
    }
    assert "secret" not in str(details)


def test_batch_errors_of_old_log_rows_do_not_reach_the_service_status_answer(
    connection, database_url, monkeypatch
):
    monkeypatch.setenv("DATABASE_URL", database_url)
    old_errors = {
        "7": "RuntimeError: results of secret-paper expired",
        "8": "InvalidPassword (SQLSTATE 28P01)",
    }
    connection.execute(
        "INSERT INTO job_run (job, status, started_at, finished_at, details)"
        " VALUES ('extract', 'failed', '2026-11-12T06:00Z', '2026-11-12T06:00Z', %s)",
        (Jsonb({"batch_errors": old_errors}),),
    )

    with TestClient(mcp_server.application(TOKEN)) as client:
        response = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {TOKEN}",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "service_status", "arguments": {}},
            },
        )

    (run,) = response.json()["result"]["structuredContent"]["latest_runs"]
    assert run["details"]["batch_errors"] == {
        "7": "RuntimeError",
        "8": "InvalidPassword (SQLSTATE 28P01)",
    }
    assert "secret" not in response.text


def test_daily_details_show_what_the_head_full_run_still_misses(connection, current):
    head = start_run(connection, model=EXTRACTION_MODEL, full=True)
    assert runs.head_candidate(connection) == head
    mark_done(connection, head, "2511.00001")

    details = daily_extraction.extract_daily(connection, FakeModel)

    assert runs.run_state(connection, head).missing == 2
    assert details["full_run_missing"] == 2


def test_daily_details_have_no_missing_count_without_a_head_full_run(connection, current):
    details = daily_extraction.extract_daily(connection, FakeModel)

    assert details["full_run_missing"] is None


class PerPaperModel(FakeModel):
    def count_tokens(self, requests, *, model, config):
        super().count_tokens(requests, model=model, config=config)
        return 1000 * len(requests)


def run_full(connection, **flags):
    lines = []
    full_run.run(
        connection,
        PerPaperModel(),
        model=EXTRACTION_MODEL,
        confirm=True,
        out=lines.append,
        **flags,
    )
    return "\n".join(lines)


def test_full_run_final_message_says_to_rerun_after_collection(connection):
    add_paper(connection, "2511.00001")

    text = run_full(connection)

    last = text.splitlines()[-1]
    assert last.startswith("Done: sent 1")
    assert "collected" in last
    assert "run the command again" in last
    assert "current" in last


def test_restart_hint_does_not_claim_the_old_run_never_becomes_current():
    assert "never" not in full_run._RESTART_HINT


def test_abandon_sending_goes_through_abandon_batch(connection, monkeypatch):
    add_paper(connection, "2511.00001")
    head = start_run(connection, model=EXTRACTION_MODEL, full=True)
    stuck = add_batch(connection, head, "sending", ["2511.00001"])
    given_up = []
    real = extraction.abandon_batch

    def spy(connection, batch_id, **kwargs):
        given_up.append(batch_id)
        return real(connection, batch_id, **kwargs)

    monkeypatch.setattr(extraction, "abandon_batch", spy)

    text = run_full(connection, abandon_sending=True)

    assert given_up == [stuck]
    assert status_of(connection, stuck) == "abandoned"
    assert "abandoned" in text.lower()


def test_abandon_batch_can_be_limited_to_sending_batches(connection, current):
    batch_id = add_batch(connection, current, "submitted", ["2511.00002"])

    with pytest.raises(extraction.BatchNotAbandonable):
        extraction.abandon_batch(connection, batch_id, statuses=("sending",))

    assert status_of(connection, batch_id) == "submitted"
    assert issubclass(extraction.BatchNotAbandonable, ValueError)


def test_abandon_sending_skips_a_batch_that_became_submitted_meanwhile(connection, monkeypatch):
    add_paper(connection, "2511.00001")
    head = start_run(connection, model=EXTRACTION_MODEL, full=True)
    stuck = add_batch(connection, head, "sending", ["2511.00001"])
    real = extraction.abandon_batch

    def late(connection, batch_id, **kwargs):
        connection.execute(
            "UPDATE extraction_batch SET status = 'submitted' WHERE id = %s", (batch_id,)
        )
        return real(connection, batch_id, **kwargs)

    monkeypatch.setattr(extraction, "abandon_batch", late)

    run_full(connection, abandon_sending=True)

    assert status_of(connection, stuck) == "submitted"


def test_abandon_sending_does_not_hide_other_value_errors(connection, monkeypatch):
    add_paper(connection, "2511.00001")
    head = start_run(connection, model=EXTRACTION_MODEL, full=True)
    add_batch(connection, head, "sending", ["2511.00001"])

    def broken(connection, batch_id, **kwargs):
        raise ValueError("the connection has a transaction open")

    monkeypatch.setattr(extraction, "abandon_batch", broken)

    with pytest.raises(ValueError, match="transaction"):
        run_full(connection, abandon_sending=True)


class AbandonedWhileSending(FakeModel):
    """The batch is given up from another session while submit_batch is in flight."""

    def __init__(self, database_url, **kwargs):
        super().__init__(**kwargs)
        self.database_url = database_url

    def submit_batch(self, requests, *, model, config):
        with psycopg.connect(self.database_url, autocommit=True) as other:
            other.execute(
                "UPDATE extraction_batch SET status = 'abandoned' WHERE status = 'sending'"
            )
        return super().submit_batch(requests, model=model, config=config)


def test_a_batch_abandoned_while_it_was_sent_stays_abandoned_and_warns(
    connection, current, database_url, caplog
):
    model = AbandonedWhileSending(database_url)

    batch_id = extraction.submit(connection, model, current, ["2511.00002"])

    assert status_of(connection, batch_id) == "abandoned"
    provider = connection.execute(
        "SELECT provider_batch_id FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone()[0]
    assert provider == "batch-1"
    assert "paid for twice" in caplog.text
    assert "batch-1" in caplog.text


def test_a_batch_abandoned_while_rejected_stays_abandoned_and_counted(
    connection, current, database_url
):
    model = AbandonedWhileSending(database_url, reject_submit=True)
    with pytest.raises(extraction.BatchRejected):
        extraction.submit(connection, model, current, ["2511.00002"])

    (batch_id,) = connection.execute("SELECT id FROM extraction_batch").fetchone()
    assert status_of(connection, batch_id) == "abandoned"
    assert extraction.spent_usd(connection) > 0


def test_abandon_batch_output_for_a_submitted_batch_says_it_left(
    connection, current, cli_database, capsys
):
    batch_id = add_batch(connection, current, "submitted", ["2511.00002"])

    cli.main(["abandon-batch", str(batch_id)])

    shown = capsys.readouterr().out
    assert "provider-submitted" in shown
    assert "console" in shown
    assert "not be collected" in shown
    assert "paid for again" in shown
    assert "for a batch that was created" not in shown


def test_abandon_batch_output_for_a_sending_batch_asks_to_check_the_console(
    connection, current, cli_database, capsys
):
    batch_id = add_batch(connection, current, "sending", ["2511.00002"])

    cli.main(["abandon-batch", str(batch_id)])

    assert "for a batch that was created" in capsys.readouterr().out


def test_abandon_batch_of_a_collected_batch_does_not_claim_the_estimate_stays(
    connection, current, cli_database, capsys
):
    batch_id = add_batch(connection, current, "collected", ["2511.00002"])

    with pytest.raises(SystemExit):
        cli.main(["abandon-batch", str(batch_id)])

    assert "stays in the spent" not in capsys.readouterr().out


def test_abandon_batch_of_an_unknown_id_exits_with_the_error_text(connection, cli_database):
    with pytest.raises(SystemExit) as stop:
        cli.main(["abandon-batch", "999"])

    assert stop.value.code == "error: no extraction batch 999"


def test_batch_errors_that_are_not_strings_are_hidden():
    run = mcp_server.JobRun(
        job="extract",
        started_at="2026-11-12T06:00Z",
        finished_at=None,
        status="failed",
        details={"batch_errors": {"7": {"text": "secret"}, "8": ["secret"]}},
    )

    shown = mcp_server._without_error_text(run)

    assert shown.details["batch_errors"] == {"7": "(скрыто)", "8": "(скрыто)"}


def test_a_failing_missing_count_does_not_stop_the_sending(connection, current, monkeypatch):
    start_run(connection, model=EXTRACTION_MODEL, full=True)

    def broken(connection):
        raise psycopg.errors.DivisionByZero("boom")

    monkeypatch.setattr(runs, "head_candidate", broken)

    details = daily_extraction.extract_daily(connection, FakeModel)

    assert details["full_run_missing"] is None
    assert details["submitted"] == 2


def test_the_missing_count_is_taken_after_the_sending(connection, current, monkeypatch):
    start_run(connection, model=EXTRACTION_MODEL, full=True)
    seen = []
    real = runs.head_candidate
    details = {}

    def spy(connection):
        seen.append(len(details["submitted_batches"]))
        return real(connection)

    monkeypatch.setattr(runs, "head_candidate", spy)

    daily_extraction.extract_daily(connection, FakeModel, details=details)

    assert seen == [1]


def test_abandon_confirm_gives_up_only_the_status_it_described(
    connection, current, cli_database, monkeypatch
):
    batch_id = add_batch(connection, current, "sending", ["2511.00002"])
    real = extraction.abandon_batch

    def late(connection, batch_id, **kwargs):
        connection.execute(
            "UPDATE extraction_batch SET status = 'submitted' WHERE id = %s", (batch_id,)
        )
        return real(connection, batch_id, **kwargs)

    monkeypatch.setattr(extraction, "abandon_batch", late)

    with pytest.raises(SystemExit) as stop:
        cli.main(["abandon-batch", str(batch_id), "--confirm"])

    assert str(stop.value.code).startswith("error: ")
    assert status_of(connection, batch_id) == "submitted"


def test_batch_errors_that_are_not_a_dict_are_hidden():
    run = mcp_server.JobRun(
        job="extract",
        started_at="2026-11-12T06:00Z",
        finished_at=None,
        status="failed",
        details={"batch_errors": ["RuntimeError: secret"]},
    )

    assert mcp_server._without_error_text(run).details["batch_errors"] == "(скрыто)"


def test_abandon_batch_refuses_a_bare_string_of_statuses(connection, current):
    batch_id = add_batch(connection, current, "sending", ["2511.00002"])

    with pytest.raises(TypeError):
        extraction.abandon_batch(connection, batch_id, statuses="sending")

    assert status_of(connection, batch_id) == "sending"


def test_abandon_sending_skips_a_batch_that_is_gone(connection, monkeypatch):
    add_paper(connection, "2511.00001")
    head = start_run(connection, model=EXTRACTION_MODEL, full=True)
    add_batch(connection, head, "sending", ["2511.00001"])

    def gone(connection, batch_id, **kwargs):
        raise LookupError(f"no extraction batch {batch_id}")

    monkeypatch.setattr(extraction, "abandon_batch", gone)

    run_full(connection, abandon_sending=True)
