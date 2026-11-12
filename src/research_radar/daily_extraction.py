"""The extraction step of the daily job: collect what was sent, switch runs, send what is due.

It never starts a run (docs/adr/0004: a full run is the developer's), and it does not
recompute history: it sends only the papers the current run has not done with their
present abstract (docs/adr/0003).
"""

import os
from collections import Counter
from collections.abc import Callable
from typing import Any

import psycopg

from research_radar import extraction, runs
from research_radar.database import connect, database_url, direct_url, require_direct
from research_radar.job_log import RunSkipped, record_run
from research_radar.settings import (
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_DAILY_MAX_PAPERS,
    EXTRACTION_MAX_ATTEMPTS,
)


class TooManyPapers(Exception):
    """More papers are due than one run of the job may send."""


# The papers of the scope the current run has not done with their present abstract and has
# not out in a batch, each flagged when the run gave up on it. GIVEN_UP_SQL only lists papers
# whose attempts were made on the abstract they have now.
_DUE_SQL = (
    """
    SELECT s.arxiv_id, g.arxiv_id IS NOT NULL AS given_up
      FROM ("""
    + extraction.SCOPE_SQL
    + """) s
      JOIN paper p ON p.arxiv_id = s.arxiv_id
      LEFT JOIN ("""
    + extraction.GIVEN_UP_SQL
    + """) g ON g.arxiv_id = s.arxiv_id
     WHERE NOT EXISTS (SELECT 1 FROM paper_extraction e
                        WHERE e.run_id = %(run)s AND e.arxiv_id = s.arxiv_id
                          AND e.abstract_hash = p.abstract_hash)
       AND NOT EXISTS (SELECT 1 FROM ("""
    + extraction.IN_FLIGHT_SQL
    + """) f
                        WHERE f.arxiv_id = s.arxiv_id AND f.abstract_hash = p.abstract_hash)
     ORDER BY s.arxiv_id COLLATE "C"
"""
)


def _due(connection: psycopg.Connection, run_id: int) -> tuple[list[str], int]:
    """The papers to send, and how many the run gave up on (those are not in the list)."""
    rows = connection.execute(
        _DUE_SQL,
        {**extraction.SCOPE_PARAMS, "run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS},
    ).fetchall()
    return [arxiv_id for arxiv_id, given_up in rows if not given_up], sum(
        given_up for _, given_up in rows
    )


class _LazyClient:
    """Builds the model client on first use: a day with nothing to collect or send must not
    need the key (the client's constructor reads it)."""

    def __init__(self, factory: Callable[[], extraction.ModelClient]) -> None:
        self._factory = factory
        self._client: extraction.ModelClient | None = None

    def resolve(self) -> extraction.ModelClient:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RunSkipped("ключ Anthropic не подключён к задаче (#22)")
            self._client = self._factory()
        return self._client

    def __getattr__(self, name: str):
        return getattr(self.resolve(), name)


def _collect_all(
    connection: psycopg.Connection, client: _LazyClient, details: dict[str, Any]
) -> None:
    """Collects every batch that is out, whatever run it belongs to; one that fails is
    recorded in details["batch_errors"] and does not stop the others."""
    collected: Counter = Counter()
    ended = 0
    errors: dict[str, str] = {}
    try:
        for (batch_id,) in connection.execute(
            "SELECT id FROM extraction_batch WHERE status IN ('sending', 'submitted') ORDER BY id"
        ).fetchall():
            try:
                counts = extraction.collect(connection, client, batch_id)
            except RunSkipped:
                raise
            except Exception as error:
                errors[str(batch_id)] = f"{type(error).__name__}: {error}"
                continue
            collected += counts
            ended += any(counts[key] for key in ("extracted", "failed", "stale"))
    finally:
        details["collected"] = dict(collected)
        details["batches_collected"] = ended
        details["batch_errors"] = errors


def _submit_all(
    connection: psycopg.Connection, client: _LazyClient, run_id: int, due: list[str], details: dict
) -> None:
    parts = [due[i : i + EXTRACTION_BATCH_SIZE] for i in range(0, len(due), EXTRACTION_BATCH_SIZE)]
    for index, part in enumerate(parts):
        try:
            batch_id = extraction.submit(connection, client, run_id, part)
        except extraction.BudgetExceeded as error:
            if not details["submitted_batches"]:
                raise
            sent = len(details["submitted_batches"])
            raise extraction.BudgetExceeded(
                f"sent {sent} parts, {len(parts) - index} left: {error}"
            ) from error
        if batch_id is not None:
            details["submitted_batches"].append(batch_id)
            details["submitted"] += len(part)


def extract_daily(
    connection: psycopg.Connection,
    client_factory: Callable[[], extraction.ModelClient],
    *,
    max_papers: int = EXTRACTION_DAILY_MAX_PAPERS,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collects the sent batches, switches to a finished full run, sends the due papers.

    ``details`` is filled step by step, so a caller that keeps the dict (the job log) still
    has what was done when a later step raises. The connection must have no transaction
    open. The model client is made only when there is a batch to collect or a paper to send,
    and without a key that is RunSkipped. A batch that cannot be collected does not stop the
    rest; the step then ends with an error naming it, after the papers were sent.
    """
    details = {} if details is None else details
    details.update(
        promoted=None,
        current_run=None,
        due=0,
        gave_up=0,
        submitted=0,
        submitted_batches=[],
    )
    client = _LazyClient(client_factory)
    try:
        _collect_all(connection, client, details)
        details["promoted"] = runs.promote(connection)
        run_id = details["current_run"] = runs.current_run(connection)
        if run_id is not None:
            due, details["gave_up"] = _due(connection, run_id)
            details["due"] = len(due)
            if len(due) > max_papers:
                raise TooManyPapers(
                    f"{len(due)} papers are due, the limit is {max_papers}; if the full run "
                    "was just switched to, run the job once with --max-extract-papers"
                )
            if due:
                client.resolve()
                _submit_all(connection, client, run_id, due, details)
    finally:
        details["spent_usd"] = str(extraction.spent_usd(connection))
    if details["batch_errors"]:
        raise RuntimeError(f"batches not collected: {details['batch_errors']}")
    return details


def run(
    url: str | None = None,
    client_factory: Callable[[], extraction.ModelClient] = extraction.AnthropicModelClient,
    max_papers: int = EXTRACTION_DAILY_MAX_PAPERS,
) -> None:
    """The job's ``extract`` step, logged in job_run on its own row."""
    with connect(direct_url(url or database_url())) as connection:
        require_direct(connection)
        # The extraction steps refuse a connection with a transaction open.
        connection.autocommit = True
        with record_run(connection, "extract") as details:
            extract_daily(connection, client_factory, max_papers=max_papers, details=details)
