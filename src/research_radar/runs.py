"""The current extraction run: the one readers use, and the switch to a newer one.

A full run becomes current only when it has covered the history that was collected at the
moment it started (docs/adr/0004); until then the service keeps answering from the
previous run.
"""

import psycopg

from research_radar.settings import CATEGORIES, HISTORY_START

# The papers a run has to cover: the slices' papers inside the history, collected before the
# run was created. Later ones are the daily job's to extract, into the current run.
_SCOPE = """
    SELECT p.arxiv_id
      FROM paper p JOIN extraction_run r ON r.id = %(run)s
     WHERE p.first_version_date >= %(start)s
       AND p.collected_at <= r.created_at
       AND EXISTS (SELECT 1 FROM paper_category c
                    WHERE c.arxiv_id = p.arxiv_id AND c.category = ANY(%(categories)s))
"""

_MISSING = f"""
    SELECT s.arxiv_id FROM ({_SCOPE}) s
     WHERE NOT EXISTS (SELECT 1 FROM paper_extraction e
                        WHERE e.run_id = %(run)s AND e.arxiv_id = s.arxiv_id)
     ORDER BY s.arxiv_id COLLATE "C"
"""


def current_run(connection: psycopg.Connection) -> int | None:
    """The run readers use, or None when no run was ever switched to."""
    row = connection.execute(
        "SELECT id FROM extraction_run WHERE current_since IS NOT NULL"
        " ORDER BY current_since DESC, id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def missing_papers(connection: psycopg.Connection, run_id: int) -> list[str]:
    """The papers in the run's scope that the run has not finished, by arxiv_id.

    A paper counts as finished whatever its abstract hash is now: extracting it again after
    a change is the daily job's work.
    """
    found = connection.execute("SELECT 1 FROM extraction_run WHERE id = %s", (run_id,))
    if found.fetchone() is None:
        raise LookupError(f"no extraction run {run_id}")
    rows = connection.execute(
        _MISSING, {"run": run_id, "start": HISTORY_START, "categories": list(CATEGORIES)}
    ).fetchall()
    return [row[0] for row in rows]


def promote(connection: psycopg.Connection) -> int | None:
    """Makes the newest finished full run current; returns its id, or None if none qualifies.

    Only a full run created after the current one is considered, so an older run never
    takes the place of a newer one, and a run that is current is not switched to again.
    """
    with connection.transaction():
        # Two promotions at once would each look at the same current run.
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('current_run'))")
        candidates = connection.execute(
            """
            SELECT id FROM extraction_run
             WHERE "full" AND current_since IS NULL
               AND created_at > coalesce(
                     (SELECT created_at FROM extraction_run
                       WHERE current_since IS NOT NULL
                       ORDER BY current_since DESC, id DESC LIMIT 1),
                     '-infinity')
             ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        for (run_id,) in candidates:
            if not missing_papers(connection, run_id):
                connection.execute(
                    "UPDATE extraction_run SET current_since = now() WHERE id = %s", (run_id,)
                )
                return run_id
    return None
