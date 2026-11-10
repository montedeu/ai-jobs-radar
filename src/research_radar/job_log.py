"""The job run log: one row per run of a scheduled job, with how it ended."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import class_row
from psycopg.types.json import Jsonb
from pydantic import BaseModel


class JobRun(BaseModel):
    job: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    details: dict[str, Any]


class RunSkipped(Exception):
    """Raised inside ``record_run`` when the job has nothing to do and nothing is wrong."""


@contextmanager
def record_run(connection: psycopg.Connection, job: str) -> Iterator[dict[str, Any]]:
    """Logs a run of ``job``; the caller fills the yielded dict with what the run did.

    The row is committed as 'running' before the work starts, so a run killed by the
    platform (timeout, out of memory) still leaves a trace.
    """
    run_id = connection.execute(
        "INSERT INTO job_run (job) VALUES (%s) RETURNING id", (job,)
    ).fetchone()[0]
    connection.commit()
    details: dict[str, Any] = {}
    try:
        yield details
    except RunSkipped as reason:
        # Logged apart from failures and not re-raised: the job exits cleanly, and a
        # reader of the log is not sent looking for a fault.
        connection.rollback()
        details["skipped"] = str(reason)
        _finish(connection, run_id, "skipped", details)
        return
    except BaseException as error:
        connection.rollback()
        details["error"] = f"{type(error).__name__}: {error}"
        _finish(connection, run_id, "failed", details)
        raise
    _finish(connection, run_id, "succeeded", details)


def _finish(connection: psycopg.Connection, run_id: int, status: str, details: dict) -> None:
    connection.execute(
        "UPDATE job_run SET status = %s, finished_at = now(), details = %s WHERE id = %s",
        (status, Jsonb(details), run_id),
    )
    connection.commit()


def latest_runs(connection: psycopg.Connection) -> list[JobRun]:
    """The most recent run of each job."""
    with connection.cursor(row_factory=class_row(JobRun)) as cursor:
        return cursor.execute(
            """
            SELECT DISTINCT ON (job) job, started_at, finished_at, status, details
            FROM job_run
            ORDER BY job, started_at DESC
            """
        ).fetchall()
