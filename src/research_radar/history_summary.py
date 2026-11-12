"""What the database holds of each slice's collected history, for the service status."""

from datetime import date, datetime

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
    months: list[MonthCount]
    empty_days: list[date]
    empty_day_count: int


class HistorySummary(BaseModel):
    slices: list[SliceHistory]
    last_daily_collection_at: datetime | None
    database_size_bytes: int


def history_summary(connection: psycopg.Connection) -> HistorySummary:
    last_collection = connection.execute(
        "SELECT max(finished_at) FROM job_run WHERE job = 'collect' AND status = 'succeeded'"
    ).fetchone()[0]
    size = connection.execute("SELECT pg_database_size(current_database())").fetchone()[0]
    return HistorySummary(
        slices=[_slice_history(connection, category) for category in CATEGORIES],
        last_daily_collection_at=last_collection,
        database_size_bytes=size,
    )


def _slice_history(connection: psycopg.Connection, category: str) -> SliceHistory:
    state = connection.execute(
        "SELECT backfilled_through, updated_through FROM collection_state WHERE category = %s",
        (category,),
    ).fetchone()
    backfilled_through, updated_through = state or (None, None)
    empty_days = _empty_days(connection, category, backfilled_through)
    return SliceHistory(
        category=category,
        backfilled_through=backfilled_through,
        updated_through=updated_through,
        months=_months(connection, category, backfilled_through),
        empty_days=empty_days,
        empty_day_count=len(empty_days),
    )


def _months(
    connection: psycopg.Connection, category: str, backfilled_through: date | None
) -> list[MonthCount]:
    counts = dict(
        connection.execute(
            """
            SELECT to_char(p.first_version_date, 'YYYY-MM'), count(*)
            FROM paper p JOIN paper_category c USING (arxiv_id)
            WHERE c.category = %s AND p.first_version_date >= %s
            GROUP BY 1
            """,
            (category, HISTORY_START),
        ).fetchall()
    )
    if not counts:
        return []
    last_year, last_month = map(int, max(counts).split("-"))
    months = []
    year, month = HISTORY_START.year, HISTORY_START.month
    while (year, month) <= (last_year, last_month):
        label = f"{year:04d}-{month:02d}"
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        last_day = date(next_year, next_month, 1).toordinal() - 1
        months.append(
            MonthCount(
                month=label,
                papers=counts.get(label, 0),
                partial=backfilled_through is None or backfilled_through.toordinal() < last_day,
            )
        )
        year, month = next_year, next_month
    return months


def _empty_days(
    connection: psycopg.Connection, category: str, backfilled_through: date | None
) -> list[date]:
    if backfilled_through is None:
        return []
    rows = connection.execute(
        """
        SELECT day::date
        FROM generate_series(%s::date, %s::date, interval '1 day') AS day
        WHERE NOT EXISTS (
            SELECT 1 FROM paper p JOIN paper_category c USING (arxiv_id)
            WHERE c.category = %s AND p.first_version_date = day::date
        )
        ORDER BY day
        """,
        (HISTORY_START, backfilled_through, category),
    ).fetchall()
    return [row[0] for row in rows]
