import itertools
from decimal import Decimal

import psycopg
import pytest
from fakes import FakeModel

from research_radar import cli, database, extraction, full_run, runs
from research_radar.extraction import (
    DEFAULT_CONFIG,
    BudgetExceeded,
    RequestConfig,
    start_run,
)
from research_radar.papers import abstract_hash
from research_radar.settings import (
    EXTRACTION_CEILING_USD,
    EXTRACTION_MAX_ATTEMPTS,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    EXTRACTION_MODEL,
)

IDS = [f"2511.0000{n}" for n in range(1, 6)]
OTHER_CONFIG = RequestConfig(reasoning=False)


_CLIENTS = itertools.count()


class PerPaperModel(FakeModel):
    """Counts 1000 input tokens per paper, as the fixture of the ticket says."""

    def __init__(self, *args, events=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events if events is not None else []

    def count_tokens(self, requests, *, model, config):
        super().count_tokens(requests, model=model, config=config)
        return 1000 * len(requests)

    def submit_batch(self, requests, *, model, config):
        self.events.append("submit_batch")
        provider_id = super().submit_batch(requests, model=model, config=config)
        # Several clients share one database in a test, and provider ids are unique there.
        return f"{provider_id}-{next(_CLIENTS)}"


class FailsOnSecondSubmit(PerPaperModel):
    def submit_batch(self, requests, *, model, config):
        if len(self.batches) == 1:
            raise RuntimeError("connection lost")
        return super().submit_batch(requests, model=model, config=config)


@pytest.fixture
def connection(database_url):
    # Autocommit: the extraction steps need a connection with no transaction open.
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


@pytest.fixture(autouse=True)
def small_batches(monkeypatch):
    monkeypatch.setattr(full_run, "EXTRACTION_BATCH_SIZE", 2)


def add_paper(connection, arxiv_id, *, abstract="a"):
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


@pytest.fixture
def papers(connection):
    for arxiv_id in IDS:
        add_paper(connection, arxiv_id)


def go(
    connection,
    client,
    *,
    confirm=True,
    config=DEFAULT_CONFIG,
    model=EXTRACTION_MODEL,
    restart=False,
    abandon_sending=False,
):
    lines = []
    plan = full_run.run(
        connection,
        client,
        model=model,
        config=config,
        confirm=confirm,
        restart=restart,
        abandon_sending=abandon_sending,
        out=lines.append,
    )
    return plan, "\n".join(lines)


def count(connection, table):
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def newest_run(connection):
    return connection.execute("SELECT max(id) FROM extraction_run").fetchone()[0]


def sent_ids(client):
    return [request.arxiv_id for batch in client.sent for request in batch]


def spend(connection, usd):
    """Money spent by an earlier run: a collected batch of a pilot run; returns its id."""
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    return connection.execute(
        "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd, actual_cost_usd)"
        " VALUES (%s, 'collected', 0, %s) RETURNING id",
        (run_id, usd),
    ).fetchone()[0]


def part_cost(papers):
    return extraction.estimate_usd(
        EXTRACTION_MODEL, 1000 * papers, papers * EXTRACTION_MAX_OUTPUT_TOKENS
    )


def mark_done(connection, run_id, *arxiv_ids):
    for arxiv_id in arxiv_ids:
        connection.execute(
            "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
            " output_tokens) VALUES (%s, %s, 'h', 1, 1)",
            (run_id, arxiv_id),
        )


def give_up_on(connection, run_id, arxiv_id):
    for _ in range(EXTRACTION_MAX_ATTEMPTS):
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'collected', 0) RETURNING id",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash, outcome)"
            " SELECT %s, arxiv_id, abstract_hash, 'invalid' FROM paper WHERE arxiv_id = %s",
            (batch_id, arxiv_id),
        )


def test_without_confirm_nothing_is_sent_and_the_reason_is_printed(connection, papers):
    client = PerPaperModel()

    plan, text = go(connection, client, confirm=False)

    assert "submit_batch" not in client.calls
    assert count(connection, "extraction_run") == 0
    assert count(connection, "extraction_batch") == 0
    assert plan.papers == 5
    assert "estimated" in text
    assert "--confirm" in text
    assert "nothing was sent" in text.lower()


def test_plan_covers_the_whole_collected_history(connection, papers):
    plan = full_run.plan(connection, PerPaperModel(), model=EXTRACTION_MODEL)

    assert plan.run_id is None
    assert plan.papers == 5
    assert plan.batches == 3
    assert plan.estimated_usd == part_cost(5)
    assert plan.spent_usd == 0
    assert plan.ceiling_usd == EXTRACTION_CEILING_USD
    assert count(connection, "extraction_run") == 0


def test_command_has_no_option_to_send_part_of_the_history(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["full-run", "--since", "2026-01-01"])

    assert stop.value.code != 0
    assert "--since" in capsys.readouterr().err


def test_estimate_is_printed_before_anything_is_sent(connection, papers):
    events = []
    client = PerPaperModel(events=events)

    full_run.run(
        connection,
        client,
        model=EXTRACTION_MODEL,
        confirm=True,
        out=lambda line: events.append(f"out: {line}"),
    )

    before_send = events[: events.index("submit_batch")]
    assert any("estimated" in event and f"{part_cost(5):.4f}" in event for event in before_send)


def test_full_run_over_the_ceiling_sends_nothing(connection, papers):
    spend(connection, EXTRACTION_CEILING_USD)
    client = PerPaperModel()

    with pytest.raises(BudgetExceeded):
        go(connection, client)

    assert "submit_batch" not in client.calls
    assert count(connection, "extraction_batch") == 1  # the one that was spent


def test_parts_sent_before_the_ceiling_stay_and_the_rest_is_not_sent(connection, papers):
    spend(connection, EXTRACTION_CEILING_USD - part_cost(2) * Decimal("1.5"))
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert sent_ids(client) == IDS[:2]
    assert count(connection, "extraction_batch") == 2  # the spent one and the sent part
    assert "sent 1, skipped 0, left 2" in text
    assert plan.batches == 3


def test_history_is_split_into_batches_of_the_configured_size(connection, papers):
    client = PerPaperModel()

    go(connection, client)

    assert [[r.arxiv_id for r in batch] for batch in client.sent] == [IDS[:2], IDS[2:4], IDS[4:]]
    assert connection.execute("SELECT is_full FROM extraction_run").fetchone()[0] is True
    assert count(connection, "extraction_batch") == 3


def test_interrupted_full_run_is_continued_not_restarted(connection, papers):
    spent = spend(connection, EXTRACTION_CEILING_USD - part_cost(2) * Decimal("1.5"))
    go(connection, PerPaperModel())
    runs_before = count(connection, "extraction_run")
    connection.execute("UPDATE extraction_batch SET actual_cost_usd = 0 WHERE id = %s", (spent,))
    second = PerPaperModel()

    plan, _ = go(connection, second)

    assert plan.run_id is not None
    assert count(connection, "extraction_run") == runs_before
    assert sent_ids(second) == IDS[2:]


def test_papers_in_flight_are_not_sent_again(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    extraction.submit(connection, PerPaperModel(), run_id, IDS[:2])
    client = PerPaperModel()

    plan, text = go(connection, client, confirm=False)

    assert (plan.run_id, plan.papers, plan.in_flight) == (run_id, 3, 2)
    assert "in flight" in text
    go(connection, client)
    assert sent_ids(client) == IDS[2:]


def test_full_run_with_another_request_config_starts_a_new_run(connection, papers):
    go(connection, PerPaperModel())
    first = connection.execute("SELECT id FROM extraction_run").fetchone()[0]
    mark_done(connection, first, *IDS)
    assert runs.promote(connection) == first
    client = PerPaperModel()

    plan, _ = go(connection, client, config=OTHER_CONFIG)

    assert plan.run_id == newest_run(connection)
    assert count(connection, "extraction_run") == 2
    assert sent_ids(client) == IDS
    assert OTHER_CONFIG in client.configs
    assert DEFAULT_CONFIG not in client.configs


def test_full_run_with_the_current_runs_config_sends_nothing(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, *IDS)
    assert runs.promote(connection) == run_id
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.papers == 0
    assert "already current" in text
    assert "nothing to send" in text
    assert client.calls == []
    assert count(connection, "extraction_run") == 1


def test_run_older_than_the_current_one_is_not_continued(connection, papers):
    older = start_run(connection, model=EXTRACTION_MODEL, full=True)
    newer = start_run(connection, model=EXTRACTION_MODEL, config=OTHER_CONFIG, full=True)
    mark_done(connection, newer, *IDS)
    assert runs.promote(connection) == newer
    client = PerPaperModel()

    plan, _ = go(connection, client)

    assert plan.run_id == newest_run(connection)
    assert count(connection, "extraction_run") == 3
    created = connection.execute("SELECT max(id) FROM extraction_run").fetchone()[0]
    assert created not in (older, newer)
    assert sent_ids(client) == IDS


def test_run_with_everything_marked_is_not_started_again(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, *IDS)
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.papers == 0
    assert "nothing to send" in text
    assert client.calls == []
    assert count(connection, "extraction_run") == 1


def test_run_blocked_by_given_up_papers_says_so(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, *IDS[2:])
    for arxiv_id in IDS[:2]:
        give_up_on(connection, run_id, arxiv_id)
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.papers == 0
    assert plan.given_up == 2
    assert "will not become current" in text
    assert "2 of 5" in text
    assert "nothing to send" not in text
    assert count(connection, "extraction_run") == 1


def test_empty_history_is_refused_and_creates_nothing(connection):
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.papers == 0
    assert "no collected history" in text
    assert client.calls == []
    assert count(connection, "extraction_run") == 0


def test_model_without_batch_prices_fails_in_the_plan_before_any_run(connection, papers):
    with pytest.raises(ValueError, match="no batch prices"):
        go(connection, PerPaperModel(), model="claude-unpriced")

    assert count(connection, "extraction_run") == 0


def test_client_error_keeps_the_parts_sent_and_the_stuck_part_stops_a_rerun(connection, papers):
    client = FailsOnSecondSubmit()
    with pytest.raises(RuntimeError, match="connection lost"):
        go(connection, client)
    assert sent_ids(client) == IDS[:2]
    second = PerPaperModel()

    plan, text = go(connection, second)

    # The part whose send failed is 'sending': it may have left, so nothing more is paid for.
    assert plan.status == full_run.STUCK_SENDING
    assert plan.refused
    assert "--restart" in text
    assert second.calls == []
    assert count(connection, "extraction_run") == 1


@pytest.fixture
def cli_client(monkeypatch, database_url):
    monkeypatch.setenv("DATABASE_URL", database_url)
    built = []

    def build():
        client = PerPaperModel()
        built.append(client)
        return client

    monkeypatch.setattr(extraction, "AnthropicModelClient", build)
    return built


def test_cli_without_confirm_counts_tokens_but_sends_nothing_and_creates_no_run(
    cli_client, papers, connection, capsys
):
    cli.main(["full-run"])

    (client,) = cli_client
    assert client.calls == ["count_tokens"]
    assert count(connection, "extraction_run") == 0
    assert "--confirm" in capsys.readouterr().out


def test_cli_with_confirm_sends_the_history(cli_client, papers, connection):
    cli.main(["full-run", "--confirm"])

    (client,) = cli_client
    assert sent_ids(client) == IDS


def test_head_run_with_another_config_is_not_continued_and_nothing_is_created(connection, papers):
    head = start_run(connection, model=EXTRACTION_MODEL, config=OTHER_CONFIG, full=True)
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.status == full_run.NEWER_RUN_OTHER_CONFIG
    assert plan.refused
    assert f"newer unfinished full run {head} with another configuration" in text
    assert client.calls == []
    assert count(connection, "extraction_run") == 1


def test_cli_exits_with_an_error_when_the_head_run_has_another_config(
    cli_client, papers, connection
):
    start_run(connection, model=EXTRACTION_MODEL, config=OTHER_CONFIG, full=True)

    with pytest.raises(SystemExit) as stop:
        cli.main(["full-run", "--confirm"])

    assert stop.value.code == 1


def test_blocked_run_is_not_paid_for_even_with_papers_missing(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    for arxiv_id in IDS[:2]:
        give_up_on(connection, run_id, arxiv_id)
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.status == full_run.BLOCKED
    assert plan.refused
    assert "will not become current" in text
    assert "--restart" in text
    assert "submit_batch" not in client.calls
    assert count(connection, "extraction_run") == 1


def test_restart_starts_a_new_run_past_the_head_of_the_queue(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    for arxiv_id in IDS[:2]:
        give_up_on(connection, run_id, arxiv_id)
    client = PerPaperModel()

    plan, _ = go(connection, client, restart=True)

    assert plan.run_id == newest_run(connection)
    assert count(connection, "extraction_run") == 2
    assert sent_ids(client) == IDS
    assert runs.head_candidate(connection) != run_id


def test_restart_needs_confirm(cli_client, papers, connection, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["full-run", "--restart"])

    assert stop.value.code != 0
    assert "--confirm" in capsys.readouterr().err
    assert count(connection, "extraction_run") == 0


def test_papers_all_stuck_in_sending_say_what_to_do(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, *IDS[2:])
    with pytest.raises(RuntimeError):
        extraction.submit(connection, PerPaperModel(fail_submit=True), run_id, IDS[:2])
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert plan.status == full_run.STUCK_SENDING
    assert plan.sending == 1
    assert "sending" in text
    assert "--restart" in text
    assert client.calls == []


def test_second_command_at_once_does_nothing(connection, papers, database_url):
    with psycopg.connect(database_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(hashtext('full_run'))")
        client = PerPaperModel()

        plan, text = go(connection, client)

        assert plan.status == full_run.LOCKED
        assert plan.refused
        assert "another full-run command is already running" in text.lower()
        assert client.calls == []
        assert count(connection, "extraction_run") == 0
    # The lock of a finished command is released: a command goes through.
    go(connection, PerPaperModel(), confirm=False)
    with psycopg.connect(database_url, autocommit=True) as other:
        assert other.execute("SELECT pg_try_advisory_lock(hashtext('full_run'))").fetchone()[0]


def test_part_with_nothing_left_to_send_is_skipped_and_counted(connection, papers, monkeypatch):
    real = extraction.submit
    calls = []

    def submit(*args, **kwargs):
        calls.append(1)
        return None if len(calls) == 1 else real(*args, **kwargs)

    monkeypatch.setattr(extraction, "submit", submit)

    _, text = go(connection, PerPaperModel())

    assert "sent 2, skipped 1, left 0" in text


def test_ceiling_after_skipped_parts_is_a_stop_not_an_error(connection, papers, monkeypatch):
    calls = []

    def submit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return None
        raise BudgetExceeded("over")

    monkeypatch.setattr(extraction, "submit", submit)

    _, text = go(connection, PerPaperModel())

    assert "Stopped at the ceiling" in text
    assert "sent 0, skipped 1, left 2" in text


def test_the_actual_number_of_papers_is_printed_after_the_run_is_created(connection, papers):
    _, text = go(connection, PerPaperModel())

    assert "Sending 5 papers in 3 parts" in text
    assert "slightly higher" in text


def test_restart_beats_an_already_current_run(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run_id, *IDS)
    assert runs.promote(connection) == run_id
    client = PerPaperModel()

    plan, text = go(connection, client, confirm=False)
    assert plan.status == full_run.ALREADY_CURRENT
    assert "--restart" in text

    plan, _ = go(connection, client, restart=True)

    assert plan.run_id == newest_run(connection)
    assert count(connection, "extraction_run") == 2
    assert sent_ids(client) == IDS


def test_a_run_blocked_between_parts_stops_the_rest(connection, papers, monkeypatch):
    real = extraction.submit
    calls = []

    def submit(connection, client, run_id, part, **kwargs):
        calls.append(part)
        result = real(connection, client, run_id, part, **kwargs)
        if len(calls) == 1:
            for arxiv_id in IDS[2:4]:
                give_up_on(connection, run_id, arxiv_id)
        return result

    monkeypatch.setattr(extraction, "submit", submit)
    client = PerPaperModel()

    plan, text = go(connection, client)

    assert len(calls) == 1
    assert plan.status == full_run.BLOCKED
    assert plan.refused
    assert "will not become current" in text
    assert "left 2" in text


def test_a_batch_stuck_in_sending_between_parts_stops_the_rest(connection, papers, monkeypatch):
    real = extraction.submit
    calls = []

    def submit(connection, client, run_id, part, **kwargs):
        calls.append(part)
        result = real(connection, client, run_id, part, **kwargs)
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'sending', 0) RETURNING id",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash)"
            " SELECT %s, arxiv_id, abstract_hash FROM paper WHERE arxiv_id = %s",
            (batch_id, IDS[4]),
        )
        return result

    monkeypatch.setattr(extraction, "submit", submit)

    plan, text = go(connection, PerPaperModel())

    assert len(calls) == 1
    assert plan.status == full_run.STUCK_SENDING
    assert "--restart" in text
    assert "left 2" in text


def test_releasing_the_lock_does_not_hide_the_original_error(connection, papers, monkeypatch):
    def submit(connection, *args, **kwargs):
        connection.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(extraction, "submit", submit)

    with pytest.raises(RuntimeError, match="boom"):
        go(connection, PerPaperModel())


def test_command_refuses_a_pooled_connection_before_doing_anything(connection, monkeypatch):
    class Pooled:
        host = "ep-example-pooler.eu-central-1.aws.neon.tech"

    monkeypatch.setattr(psycopg.Connection, "info", Pooled(), raising=False)
    client = PerPaperModel()

    with pytest.raises(RuntimeError, match="direct URL"):
        full_run.run(connection, client, model=EXTRACTION_MODEL, confirm=True)

    assert client.calls == []


def test_cli_connects_to_the_direct_address_of_a_pooled_url(
    monkeypatch, cli_client, database_url, papers
):
    asked = []
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@ep-a-1-pooler.x.neon.tech/db")

    def connect(url):
        asked.append(url)
        return psycopg.connect(database_url)

    monkeypatch.setattr(database, "connect", connect)

    cli.main(["full-run"])

    assert len(asked) == 1
    assert "ep-a-1.x.neon.tech" in asked[0] and "pooler" not in asked[0]


def test_restarted_run_stopped_at_the_ceiling_is_continued_by_the_next_plain_command(
    connection, papers
):
    current = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, current, *IDS)
    assert runs.promote(connection) == current
    spent = spend(connection, EXTRACTION_CEILING_USD - part_cost(2) * Decimal("1.5"))
    go(connection, PerPaperModel(), restart=True)
    restarted = newest_run(connection)
    runs_before = count(connection, "extraction_run")
    connection.execute("UPDATE extraction_batch SET actual_cost_usd = 0 WHERE id = %s", (spent,))
    client = PerPaperModel()

    plan, _ = go(connection, client)

    assert plan.run_id == restarted
    assert count(connection, "extraction_run") == runs_before
    assert sent_ids(client) == IDS[2:]


def stuck_run(connection):
    """A run whose second part was left in 'sending'; returns the run id."""
    with pytest.raises(RuntimeError, match="connection lost"):
        go(connection, FailsOnSecondSubmit())
    return newest_run(connection)


def test_abandoning_the_sending_batch_continues_the_run_and_keeps_its_cost(connection, papers):
    run_id = stuck_run(connection)
    before = extraction.spent_usd(connection)
    client = PerPaperModel()

    plan, text = go(connection, client, abandon_sending=True)

    assert sent_ids(client) == IDS[2:]
    assert plan.run_id == run_id
    assert (
        connection.execute(
            "SELECT count(*) FROM extraction_batch WHERE run_id = %s AND status = 'abandoned'",
            (run_id,),
        ).fetchone()[0]
        == 1
    )
    assert extraction.spent_usd(connection) >= before
    assert "abandoned" in text.lower()


def test_an_abandoned_batch_counts_as_spent_at_its_estimate(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    connection.execute(
        "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
        " VALUES (%s, 'abandoned', 1.5)",
        (run_id,),
    )

    assert extraction.spent_usd(connection) == Decimal("1.5")


def test_collect_leaves_an_abandoned_batch_alone(connection, papers):
    run_id = stuck_run(connection)
    go(connection, PerPaperModel(), abandon_sending=True)
    batch_id = connection.execute(
        "SELECT id FROM extraction_batch WHERE run_id = %s AND status = 'abandoned'", (run_id,)
    ).fetchone()[0]
    client = PerPaperModel()

    result = extraction.collect(connection, client, batch_id)

    assert client.calls == []
    assert result == {"abandoned": 2}
    status = "SELECT status FROM extraction_batch WHERE id = %s"
    assert connection.execute(status, (batch_id,)).fetchone()[0] == "abandoned"


def test_abandon_sending_needs_confirm_and_excludes_restart(cli_client, papers, capsys):
    for flags in (["--abandon-sending"], ["--confirm", "--abandon-sending", "--restart"]):
        with pytest.raises(SystemExit) as stop:
            cli.main(["full-run", *flags])
        assert stop.value.code != 0


def test_stuck_hint_names_the_abandon_flag_and_warns_about_the_console(connection, papers):
    stuck_run(connection)

    _, text = go(connection, PerPaperModel())

    assert "--abandon-sending" in text
    assert "console" in text
    assert "stays in the spent" in text


def test_blocked_and_stuck_run_reports_blocked_first(connection, papers):
    run_id = stuck_run(connection)
    for arxiv_id in IDS[:2]:
        give_up_on(connection, run_id, arxiv_id)

    plan, text = go(connection, PerPaperModel())

    assert plan.status == full_run.BLOCKED
    assert text.index("will not become current") < text.index("sending")
    assert "--abandon-sending" in text


def test_blocked_run_without_sending_batches_does_not_talk_about_sending(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    for arxiv_id in IDS[:2]:
        give_up_on(connection, run_id, arxiv_id)

    _, text = go(connection, PerPaperModel())

    assert "sending" not in text
    assert "--restart" in text


def test_sending_batches_are_counted_by_batch_not_by_paper_rows(connection, papers):
    run_id = start_run(connection, model=EXTRACTION_MODEL, full=True)
    connection.execute(
        "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
        " VALUES (%s, 'sending', 0)",
        (run_id,),
    )

    plan, _ = go(connection, PerPaperModel())

    assert (plan.status, plan.sending) == (full_run.STUCK_SENDING, 1)


def test_command_needs_an_autocommit_connection(database_url, papers):
    with psycopg.connect(database_url) as plain:
        client = PerPaperModel()

        with pytest.raises(ValueError, match="autocommit"):
            full_run.run(plain, client, model=EXTRACTION_MODEL, confirm=False)

    assert client.calls == []


@pytest.mark.parametrize(
    ("host", "options"),
    [
        ("EP-Example-POOLER.eu-central-1.aws.neon.tech", ""),
        ("db.example.com", "endpoint=ep-example-pooler"),
    ],
)
def test_pooled_connection_is_recognised_in_any_case_and_in_options(
    connection, monkeypatch, host, options
):
    class Pooled:
        pass

    Pooled.host, Pooled.options = host, options
    monkeypatch.setattr(psycopg.Connection, "info", Pooled(), raising=False)

    with pytest.raises(RuntimeError, match="direct URL"):
        database.require_direct(connection)
