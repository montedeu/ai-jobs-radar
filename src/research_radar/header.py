"""The header every tool answer starts with: when it was computed, from which run and
dictionary, and what the service says about its own numbers."""

from datetime import UTC, date, datetime, timedelta

import psycopg
from pydantic import BaseModel

from research_radar.extraction import GIVEN_UP_SQL, SCOPE_PARAMS, SCOPE_SQL
from research_radar.history_summary import slice_collected_through
from research_radar.runs import current_run
from research_radar.settings import CATEGORIES, EXTRACTION_MAX_ATTEMPTS, HISTORY_START
from research_radar.synonyms import current_dictionary

# CLAUDE.md, "Что сервис оговаривает в ответах".
NOTES = [
    "«Растёт» означает «чаще упоминают в аннотациях».",
    "Иерархии сущностей нет: «PEFT» считается только по статьям, где написано «PEFT».",
    "Цифры качества относятся к cs.CL и к прочтению правил одним разметчиком.",
    "Проект независим от arXiv и им не одобрен.",
]

# Over the papers of the collected slices since HISTORY_START: the earliest one the run has
# neither finished nor given up on, and the latest one it has finished. A paper the run has
# finished counts as processed even if its abstract changed afterwards: its old mentions stay
# until the daily extraction re-runs it; if the run gave up on the new abstract, until the
# next full run. Checking the hash here would roll the boundary back to old,
# already-corrected papers almost every day.
_BOUNDS = (
    """
    SELECT min(p.first_version_date) FILTER (WHERE e.arxiv_id IS NULL AND g.arxiv_id IS NULL),
           max(p.first_version_date) FILTER (WHERE e.arxiv_id IS NOT NULL)
      FROM ("""
    + SCOPE_SQL
    + """) s
      JOIN paper p ON p.arxiv_id = s.arxiv_id
      LEFT JOIN paper_extraction e ON e.run_id = %(run)s AND e.arxiv_id = s.arxiv_id
      LEFT JOIN ("""
    + GIVEN_UP_SQL
    + """) g ON g.arxiv_id = s.arxiv_id
"""
)


class ResponseHeader(BaseModel):
    as_of: date
    extraction_run: int | None
    processed_through: date | None
    dictionary_updated_at: datetime | None
    dictionary_sha256: str | None
    notes: list[str]


def response_header(connection: psycopg.Connection, *, as_of: date | None = None) -> ResponseHeader:
    run = current_run(connection)
    as_of_day = as_of or datetime.now(UTC).date() - timedelta(days=1)
    dictionary = current_dictionary(connection)
    return ResponseHeader(
        as_of=as_of_day,
        extraction_run=run,
        processed_through=None if run is None else _processed_through(connection, run, as_of_day),
        dictionary_updated_at=dictionary.loaded_at if dictionary else None,
        dictionary_sha256=dictionary.file_sha256 if dictionary else None,
        notes=list(NOTES),
    )


def _processed_through(connection: psycopg.Connection, run: int, as_of: date) -> date | None:
    """The last day D such that the run has finished every paper of the collected slices
    dated up to D, leaving out the papers it gave up on (ADR 0004).

    Never past the collected history: a paper that is not stored yet is not finished
    either, so a slice added before its backfill would otherwise claim days it lacks. Nor
    past as_of. None when the history's first day is not finished: nothing is processed.
    """
    collected = [slice_collected_through(connection, category) for category in CATEGORIES]
    if None in collected:
        return None
    first_missing, last_finished = connection.execute(
        _BOUNDS, {**SCOPE_PARAMS, "run": run, "attempts": EXTRACTION_MAX_ATTEMPTS}
    ).fetchone()
    through = first_missing - timedelta(days=1) if first_missing else last_finished
    if through is None:
        return None
    through = min(through, as_of, *collected)
    return through if through >= HISTORY_START else None
