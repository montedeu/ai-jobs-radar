"""The current extraction run: the one readers use, and the switch to a newer one.

A full run becomes current only when it has covered the history that was collected at the
moment it started (docs/adr/0004); until then the service keeps answering from the
previous run. What it has to cover is the snapshot start_run wrote into extraction_run_scope.
"""

import psycopg

from research_radar.extraction import _require_idle
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS

# The papers of a run's snapshot that still hold the switch back: no completion mark, and
# not given up on. A paper is given up on after EXTRACTION_MAX_ATTEMPTS collected batches
# of the run carried its current abstract without a usable answer; an older abstract's
# attempts do not count, the text is another one.
_MISSING = """
    SELECT s.arxiv_id
      FROM extraction_run_scope s JOIN paper p USING (arxiv_id)
     WHERE s.run_id = %(run)s
       AND NOT EXISTS (SELECT 1 FROM paper_extraction e
                        WHERE e.run_id = s.run_id AND e.arxiv_id = s.arxiv_id)
       AND (SELECT count(*)
              FROM extraction_batch_paper bp JOIN extraction_batch b ON b.id = bp.batch_id
             WHERE b.run_id = s.run_id AND b.status = 'collected'
               AND bp.arxiv_id = s.arxiv_id AND bp.abstract_hash = p.abstract_hash
           ) < %(attempts)s
"""


def current_run(connection: psycopg.Connection) -> int | None:
    """The run readers use, or None when no run was ever switched to."""
    row = connection.execute(
        "SELECT id FROM extraction_run WHERE current_since IS NOT NULL"
        " ORDER BY current_since DESC, id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def missing_papers(connection: psycopg.Connection, run_id: int) -> list[str]:
    """The papers of the run's snapshot that keep it from becoming current, by arxiv_id.

    A paper counts as finished whatever its abstract hash is now: extracting it again after
    a change is the daily job's work.
    """
    row = connection.execute(
        f"""
        SELECT array(SELECT m.arxiv_id FROM ({_MISSING}) m ORDER BY m.arxiv_id COLLATE "C")
          FROM extraction_run r WHERE r.id = %(run)s
        """,
        {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS},
    ).fetchone()
    if row is None:
        raise LookupError(f"no extraction run {run_id}")
    return row[0]


def promote(connection: psycopg.Connection) -> int | None:
    """Makes the newest finished full run current; returns its id, or None if none qualifies.

    Only a full run created after the current one is considered, so an older run never
    takes the place of a newer one, and a run that is current is not switched to again.
    The connection must have no transaction open, as in extraction.
    """
    _require_idle(connection)
    with connection.transaction():
        # Two promotions at once would each look at the same current run.
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('current_run'))")
        candidates = connection.execute(
            """
            SELECT id FROM extraction_run
             WHERE is_full AND current_since IS NULL
               AND created_at > coalesce(
                     (SELECT created_at FROM extraction_run WHERE id = %s), '-infinity')
             ORDER BY created_at DESC, id DESC
            """,
            (current_run(connection),),
        ).fetchall()
        for (run_id,) in candidates:
            unfinished = connection.execute(
                f"SELECT EXISTS ({_MISSING})", {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS}
            ).fetchone()[0]
            if not unfinished:
                # Not now(): that is the start of the transaction, and two switches in a
                # row must still be ordered.
                connection.execute(
                    "UPDATE extraction_run SET current_since = clock_timestamp() WHERE id = %s",
                    (run_id,),
                )
                return run_id
    return None
