import itertools
from decimal import Decimal

import psycopg
import pytest
from fakes import FakeModel

from research_radar import cli, extraction, full_run, runs
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


def go(connection, client, *, confirm=True, config=DEFAULT_CONFIG, model=EXTRACTION_MODEL):
    lines = []
    plan = full_run.run(
        connection, client, model=model, config=config, confirm=confirm, out=lines.append
    )
    return plan, "\n".join(lines)


def count(connection, table):
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


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
    return extraction._usd(EXTRACTION_MODEL, 1000 * papers, papers * EXTRACTION_MAX_OUTPUT_TOKENS)


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
    assert "1 of 3" in text
    assert "2 left" in text
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
    client = PerPaperModel()

    plan, _ = go(connection, client, config=OTHER_CONFIG)

    assert plan.run_id is None
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

    assert plan.run_id is None
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


def test_client_error_keeps_the_parts_already_sent_and_a_rerun_continues(connection, papers):
    client = FailsOnSecondSubmit()
    with pytest.raises(RuntimeError, match="connection lost"):
        go(connection, client)
    assert sent_ids(client) == IDS[:2]
    second = PerPaperModel()

    go(connection, second)

    assert count(connection, "extraction_run") == 1
    # The part whose send failed was recorded as 'sending' and may have left: not sent again.
    assert sent_ids(second) == IDS[4:]


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
