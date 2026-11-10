"""Collects Papers from arXiv: the history once, then every day what changed.

Both jobs are safe to repeat. A Paper is stored by its arXiv id, so reading one twice
changes nothing, and each job records how far it got only after committing that work.
"""

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from functools import partial

import psycopg

from research_radar import papers
from research_radar.arxiv_api import ArxivClient, Paper
from research_radar.database import connect
from research_radar.job_log import RunSkipped, record_run
from research_radar.settings import CATEGORIES, HISTORY_START

# How far before the previous run's mark the daily job re-reads. arXiv publishes in
# batches and a batch can still be growing while it is read; the overlap costs one
# more page and nothing else.
UPDATE_OVERLAP = timedelta(days=1)
# With no mark yet (a slice just added), the daily job reads this far back; older
# papers are the backfill's.
FIRST_RUN_LOOKBACK = timedelta(days=7)
# A normal day is one or two pages. Past this many papers something is wrong (arXiv
# re-dated its records, or the mark was lost), and stopping is cheaper than a wrong guess.
MAX_PAPERS_PER_DAILY_RUN = 5000
# arXiv finishes publishing a day's submissions later; a day this recent may be incomplete.
BACKFILL_LAG_DAYS = 2
# An arbitrary constant: the key of the Postgres advisory lock shared by both jobs.
COLLECTION_LOCK = 20251101


def _state(connection: psycopg.Connection, category: str) -> tuple[date | None, datetime | None]:
    connection.execute(
        "INSERT INTO collection_state (category) VALUES (%s) ON CONFLICT DO NOTHING", (category,)
    )
    return connection.execute(
        "SELECT backfilled_through, updated_through FROM collection_state WHERE category = %s",
        (category,),
    ).fetchone()


def _store_all(connection: psycopg.Connection, batch: list[Paper], counts: Counter) -> None:
    for paper in batch:
        if paper.first_version_date < HISTORY_START:
            counts["before_history"] += 1
        else:
            counts[papers.store(connection, paper)] += 1
        counts["received"] += 1


def collect_updates(
    connection: psycopg.Connection,
    client: ArxivClient,
    category: str,
    *,
    max_papers: int | None = None,
) -> Counter:
    """Stores every Paper of the slice that is new or has a new version since the last run."""
    _, updated_through = _state(connection, category)
    connection.commit()
    now = datetime.now(UTC)
    read_back_to = (updated_through or now - FIRST_RUN_LOOKBACK) - UPDATE_OVERLAP

    limit = MAX_PAPERS_PER_DAILY_RUN if max_papers is None else max_papers
    counts: Counter = Counter()
    batch: list[Paper] = []
    newest: datetime | None = None
    for paper in client.latest_updated(category):
        if paper.arxiv_updated_at < read_back_to:
            break
        newest = newest or paper.arxiv_updated_at
        batch.append(paper)
        if len(batch) > limit:
            raise RuntimeError(
                f"{category}: more than {limit} papers "
                f"updated since {read_back_to:%Y-%m-%d %H:%M}; refusing to continue. If "
                "the job was simply down for a long time, run it once with --max-papers"
            )
    # One transaction for the papers and the mark: a run that fails halfway must leave
    # the mark where it was, or the papers it did not store would never be read.
    _store_all(connection, batch, counts)
    if newest:
        connection.execute(
            "UPDATE collection_state SET updated_through = %s WHERE category = %s",
            (newest, category),
        )
    connection.commit()
    return counts


def backfill(
    connection: psycopg.Connection,
    client: ArxivClient,
    category: str,
    *,
    today: date | None = None,
) -> Counter:
    """Stores the slice's history day by day, from where the previous run stopped."""
    backfilled_through, _ = _state(connection, category)
    connection.commit()
    last_day = (today or datetime.now(UTC).date()) - timedelta(days=BACKFILL_LAG_DAYS)
    day = backfilled_through + timedelta(days=1) if backfilled_through else HISTORY_START

    counts: Counter = Counter()
    while day <= last_day:
        _store_all(connection, list(client.submitted_on(category, day)), counts)
        connection.execute(
            "UPDATE collection_state SET backfilled_through = %s WHERE category = %s",
            (day, category),
        )
        # Per day: a run cut short by the job's time limit keeps what it finished.
        connection.commit()
        counts["days"] += 1
        day += timedelta(days=1)
    return counts


def _run(job: str, work, url: str | None, client: ArxivClient | None) -> None:
    client = client or ArxivClient()
    with connect(url) as connection, record_run(connection, job) as details:
        # arXiv allows one connection for everything this project runs, and the daily
        # job and the backfill lock the same rows in opposite orders. The lock is held
        # by the session and goes when the connection closes; behind a pooler a session
        # is not ours, and the lock would be handed to whoever gets the backend next.
        if "-pooler" in (connection.info.host or ""):
            raise RuntimeError("DATABASE_URL points at Neon's pooler; jobs need the direct URL")
        (alone,) = connection.execute(
            "SELECT pg_try_advisory_lock(%s)", (COLLECTION_LOCK,)
        ).fetchone()
        if not alone:
            raise RunSkipped("another collection job is running")
        for category in CATEGORIES:
            try:
                details[category] = dict(work(connection, client, category))
            finally:
                details["requests"] = client.requests


def run(
    url: str | None = None, client: ArxivClient | None = None, max_papers: int | None = None
) -> None:
    _run("collect", partial(collect_updates, max_papers=max_papers), url, client)


def run_backfill(url: str | None = None, client: ArxivClient | None = None) -> None:
    _run("backfill", backfill, url, client)
