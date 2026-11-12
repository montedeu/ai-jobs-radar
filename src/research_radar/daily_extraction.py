"""The extraction step of the daily job: collect what was sent, switch runs, send what is due.

It never starts a run (docs/adr/0004: a full run is the developer's), and it does not
recompute history: it sends only the papers the current run has not done with their
present abstract (docs/adr/0003).
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import psycopg

from research_radar import extraction, runs
from research_radar.database import connect, database_url, direct_url, require_direct
from research_radar.job_log import record_run
from research_radar.settings import EXTRACTION_DAILY_MAX_PAPERS, EXTRACTION_MAX_ATTEMPTS


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
    open. The model client is made only when there is a batch to collect or a paper to send.
    """
    details = {} if details is None else details
    client: extraction.ModelClient | None = None

    def get_client() -> extraction.ModelClient:
        nonlocal client
        client = client or client_factory()
        return client

    batches = connection.execute(
        "SELECT id, status FROM extraction_batch WHERE status IN ('sending', 'submitted')"
        " ORDER BY id"
    ).fetchall()
    collected: Counter = Counter()
    details["collected"] = {}
    details["batches_collected"] = 0
    for batch_id, status in batches:
        if status == "sending":
            # Nobody knows whether it left; collect would only count it, so no client is made.
            collected["unsent"] += connection.execute(
                "SELECT count(*) FROM extraction_batch_paper WHERE batch_id = %s", (batch_id,)
            ).fetchone()[0]
            continue
        collected += extraction.collect(connection, get_client(), batch_id)
        details["batches_collected"] += 1
        details["collected"] = dict(collected)
    details["collected"] = dict(collected)
    details["spent_usd"] = str(extraction.spent_usd(connection))

    details["promoted"] = runs.promote(connection)
    run_id = runs.current_run(connection)
    details["current_run"] = run_id
    details.update(due=0, gave_up=0, submitted=0, submitted_batch=None)
    if run_id is None:
        return details

    due, details["gave_up"] = _due(connection, run_id)
    details["due"] = len(due)
    if len(due) > max_papers:
        raise TooManyPapers(
            f"{len(due)} papers are due, the limit is {max_papers}; if the full run was "
            "just switched to, run the job once with --max-extract-papers"
        )
    if not due:
        return details

    batch_id = extraction.submit(connection, get_client(), run_id, due)
    details["submitted_batch"] = batch_id
    if batch_id is not None:
        details["submitted"] = connection.execute(
            "SELECT count(*) FROM extraction_batch_paper WHERE batch_id = %s", (batch_id,)
        ).fetchone()[0]
    details["spent_usd"] = str(extraction.spent_usd(connection))
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
