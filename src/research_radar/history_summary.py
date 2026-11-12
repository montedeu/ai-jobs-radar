"""What the database holds of each slice's collected history, for the service status."""

from datetime import UTC, date, datetime, timedelta

import psycopg
from pydantic import BaseModel

from research_radar.settings import CATEGORIES, HISTORY_START


class MonthCount(BaseModel):
    month: str
    papers: int
    # The month is not wholly inside the collected history, so its count may be short.
    partial: bool


class SliceHistory(BaseModel):
    category: str
    backfilled_through: date | None
    updated_through: datetime | None
    # Every day up to this one is collected: the backfill boundary, or the later of it and the
    # day before updated_through once the last backfill succeeded. None until the backfill
    # has set the boundary.
    collected_through: date | None
    months: list[MonthCount]
    empty_days: list[date]
    empty_day_count: int


class HistorySummary(BaseModel):
    slices: list[SliceHistory]
    last_daily_collection_at: datetime | None
    database_size_bytes: int


def history_summary(connection: psycopg.Connection) -> HistorySummary:
    """Reads everything from one snapshot, so the slices and the size agree with each other.

    The connection must have no transaction open: the isolation level cannot change inside one.
    """
    connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    connection.read_only = True
    try:
        last_collection = connection.execute(
            "SELECT max(finished_at) FROM job_run WHERE job = 'collect' AND status = 'succeeded'"
        ).fetchone()[0]
        slices = [_slice_history(connection, category) for category in CATEGORIES]
        size = connection.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        connection.commit()
    finally:
        # After an error the transaction is still open and the settings cannot change in
        # it; the rollback keeps the caller seeing the original error. After a commit it
        # does nothing.
        connection.rollback()
        connection.isolation_level = None
        connection.read_only = None
    return HistorySummary(
        slices=slices, last_daily_collection_at=last_collection, database_size_bytes=size
    )


def _slice_history(connection: psycopg.Connection, category: str) -> SliceHistory:
    state = connection.execute(
        "SELECT backfilled_through, updated_through FROM collection_state WHERE category = %s",
        (category,),
    ).fetchone()
    backfilled_through, updated_through = state or (None, None)
    last_backfill = connection.execute(
        "SELECT status FROM job_run WHERE job = 'backfill' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    backfill_succeeded = last_backfill is not None and last_backfill[0] == "succeeded"
    collected_through = _collected_through(backfilled_through, updated_through, backfill_succeeded)
    papers_per_day = dict(
        connection.execute(
            """
            SELECT p.first_version_date, count(*)
            FROM paper p JOIN paper_category c USING (arxiv_id)
            WHERE c.category = %s AND p.first_version_date >= %s
            GROUP BY 1
            """,
            (category, HISTORY_START),
        ).fetchall()
    )
    empty_days = _empty_days(papers_per_day, collected_through)
    return SliceHistory(
        category=category,
        backfilled_through=backfilled_through,
        updated_through=updated_through,
        collected_through=collected_through,
        months=_months(papers_per_day, collected_through),
        empty_days=empty_days,
        empty_day_count=len(empty_days),
    )


def _collected_through(
    backfilled_through: date | None, updated_through: datetime | None, backfill_succeeded: bool
) -> date | None:
    # Without a backfill boundary the history before the daily job's own reach is unknown,
    # so no day is claimed as collected.
    if backfilled_through is None:
        return None
    # The boundary moves day by day from the start, so it also stands after a backfill that
    # stopped midway. The daily job's reach counts only once the last backfill succeeded.
    if updated_through is None or not backfill_succeeded:
        return backfilled_through
    # arXiv time is UTC and the daily job rereads from a day before updated_through.
    closed = updated_through.astimezone(UTC).date() - timedelta(days=1)
    return max(backfilled_through, closed)


def _months(papers_per_day: dict[date, int], collected_through: date | None) -> list[MonthCount]:
    last_days = [day for day in (max(papers_per_day, default=None), collected_through) if day]
    if not last_days:
        return []
    per_month: dict[tuple[int, int], int] = {}
    for day, papers in papers_per_day.items():
        per_month[(day.year, day.month)] = per_month.get((day.year, day.month), 0) + papers
    last = max((day.year, day.month) for day in last_days)
    months = []
    year, month = HISTORY_START.year, HISTORY_START.month
    while (year, month) <= last:
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        last_day = date(next_year, next_month, 1) - timedelta(days=1)
        months.append(
            MonthCount(
                month=f"{year:04d}-{month:02d}",
                papers=per_month.get((year, month), 0),
                partial=collected_through is None or collected_through < last_day,
            )
        )
        year, month = next_year, next_month
    return months


def _empty_days(papers_per_day: dict[date, int], collected_through: date | None) -> list[date]:
    if collected_through is None:
        return []
    days = (collected_through - HISTORY_START).days + 1
    every_day = (HISTORY_START + timedelta(days=offset) for offset in range(days))
    return [day for day in every_day if day not in papers_per_day]
