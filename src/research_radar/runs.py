"""The current extraction run: the one readers use, and the switch to a newer one.

A full run becomes current only when it has covered the history that was collected at the
moment it started (docs/adr/0004); until then the service keeps answering from the
previous run. What it has to cover is the snapshot start_run wrote into extraction_run_scope.
"""

import logging

import psycopg

from research_radar.extraction import GIVEN_UP_SQL, require_idle
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS, EXTRACTION_MAX_GIVEN_UP_SHARE

log = logging.getLogger(__name__)

# The papers of a run's snapshot that still hold the switch back: no completion mark, and
# not given up on (see extraction.GIVEN_UP_SQL).
_MISSING = (
    """
    SELECT s.arxiv_id
      FROM extraction_run_scope s
      LEFT JOIN ("""
    + GIVEN_UP_SQL
    + """) g USING (arxiv_id)
     WHERE s.run_id = %(run)s AND g.arxiv_id IS NULL
       AND NOT EXISTS (SELECT 1 FROM paper_extraction e
                        WHERE e.run_id = s.run_id AND e.arxiv_id = s.arxiv_id)
"""
)

# One pass over the scope: how many papers still hold the run back, how many it gave up
# on, and the size of the scope.
_STATE = (
    """
    SELECT count(*) FILTER (WHERE e.arxiv_id IS NULL AND g.arxiv_id IS NULL),
           count(*) FILTER (WHERE e.arxiv_id IS NULL AND g.arxiv_id IS NOT NULL),
           count(*)
      FROM extraction_run_scope s
      LEFT JOIN paper_extraction e ON e.run_id = s.run_id AND e.arxiv_id = s.arxiv_id
      LEFT JOIN ("""
    + GIVEN_UP_SQL
    + """) g ON g.arxiv_id = s.arxiv_id
     WHERE s.run_id = %(run)s
"""
)


def current_run(connection: psycopg.Connection) -> int | None:
    """The run readers use, or None when no run was ever switched to."""
    row = connection.execute(
        "SELECT id FROM extraction_run WHERE current_since IS NOT NULL"
        " ORDER BY current_since DESC, id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None


def missing_papers(connection: psycopg.Connection, run_id: int) -> list[str]:
    """The papers of the run's snapshot that keep it from becoming current, by arxiv_id.

    Papers the run gave up on are not listed. A paper counts as finished whatever its
    abstract hash is now: extracting it again after a change is the daily job's work.
    Only a full run has a scope; a pilot run raises ValueError.
    """
    row = connection.execute(
        f"""
        SELECT r.is_full,
               array(SELECT m.arxiv_id FROM ({_MISSING}) m ORDER BY m.arxiv_id COLLATE "C")
          FROM extraction_run r WHERE r.id = %(run)s
        """,
        {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS},
    ).fetchone()
    if row is None:
        raise LookupError(f"no extraction run {run_id}")
    if not row[0]:
        raise ValueError(f"run {run_id} is not a full run: only a full run has a scope")
    return row[1]


def promote(connection: psycopg.Connection) -> int | None:
    """Makes the newest finished full run current; returns its id, or None if none qualifies.

    Only a full run created after the current one is considered, so an older run never
    takes the place of a newer one, and a run that is current is not switched to again.
    A run that gave up on more than one paper and on more than EXTRACTION_MAX_GIVEN_UP_SHARE
    of its scope is not switched to: that is a failure of the run, not of a few papers.
    The connection must have no transaction open, as in extraction.
    """
    require_idle(connection)
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
            missing, given_up, scope = connection.execute(
                _STATE, {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS}
            ).fetchone()
            if missing:
                continue
            if given_up > max(1, EXTRACTION_MAX_GIVEN_UP_SHARE * scope):
                log.warning(
                    "run %s covered its scope but gave up on %s of %s papers; not switching",
                    run_id,
                    given_up,
                    scope,
                )
                continue
            # Not now(): that is the start of the transaction, and two switches in a
            # row must still be ordered.
            connection.execute(
                "UPDATE extraction_run SET current_since = clock_timestamp() WHERE id = %s",
                (run_id,),
            )
            return run_id
    return None
