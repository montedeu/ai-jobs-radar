"""Growth of entities between two adjacent windows (docs/adr/0004).

The query itself lives in ``sql/growth.sql`` and is written by the developer; this module
only runs it. See docs/growth-query.md for what it must take and return.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import psycopg
from psycopg.rows import class_row
from pydantic import BaseModel

GROWTH_SQL = Path(__file__).parent / "sql" / "growth.sql"

DEFAULT_WINDOW_DAYS = 90
DEFAULT_MIN_MENTIONS = 20
DEFAULT_NEW_MAX_PREVIOUS = 5


class EntityGrowth(BaseModel):
    canonical_name: str
    entity_type: Literal["method", "model", "task", "dataset"]
    kind: Literal["rising", "new", "fading"]
    mentions_last: int
    mentions_previous: int
    papers_last: int
    papers_previous: int
    share_last: float
    share_previous: float
    ratio: float | None


def entity_growth(
    connection: psycopg.Connection,
    *,
    run_id: int,
    category: str,
    as_of: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_mentions: int = DEFAULT_MIN_MENTIONS,
    new_max_previous: int = DEFAULT_NEW_MAX_PREVIOUS,
) -> list[EntityGrowth]:
    parameters = {
        "run_id": run_id,
        "category": category,
        "as_of": as_of,
        "window_days": window_days,
        "min_mentions": min_mentions,
        "new_max_previous": new_max_previous,
    }
    with connection.cursor(row_factory=class_row(EntityGrowth)) as cursor:
        return cursor.execute(GROWTH_SQL.read_text(), parameters).fetchall()
