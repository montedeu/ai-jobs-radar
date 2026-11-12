"""Finding papers by the text of the title and abstract, by slice and date, and by entity."""

from datetime import date

import psycopg
from pydantic import BaseModel

from research_radar.header import ResponseHeader, response_header
from research_radar.normalize import normal_form
from research_radar.runs import current_run
from research_radar.settings import CATEGORIES

# The expression of the index paper_search (migration 0010); the planner uses the index
# only for the same text.
_DOCUMENT = "to_tsvector('english', p.title || ' ' || p.abstract)"
_TEXT_QUERY = "websearch_to_tsquery('english', %(query)s)"


class BadRequest(ValueError):
    """The request cannot be meant; the message says what to change."""


class PaperHit(BaseModel):
    arxiv_id: str
    title: str
    date: date  # first_version_date
    url: str


class SearchResult(BaseModel):
    header: ResponseHeader
    papers: list[PaperHit]
    # Why the list is empty, or the request was not run; None when papers were found.
    message: str | None = None


def build_query(
    *,
    query: str | None,
    entity: str | None,
    category: str,
    date_from: date | None,
    date_to: date | None,
    limit: int,
    run: int | None = None,
) -> tuple[str, dict]:
    """The search as SQL and its parameters. `entity` is a canonical name, looked up in
    the mentions of `run`."""
    where = [
        "EXISTS (SELECT 1 FROM paper_category c"
        " WHERE c.arxiv_id = p.arxiv_id AND c.category = %(category)s)"
    ]
    params: dict = {"category": category, "limit": limit}
    order = []
    if query is not None:
        where.append(f"{_DOCUMENT} @@ {_TEXT_QUERY}")
        order.append(f"ts_rank({_DOCUMENT}, {_TEXT_QUERY}) DESC")
        params["query"] = query
    if entity is not None:
        where.append(
            "EXISTS (SELECT 1 FROM paper_entity e WHERE e.run_id = %(run)s"
            " AND e.arxiv_id = p.arxiv_id AND e.canonical_name = %(entity)s)"
        )
        params.update(run=run, entity=entity)
    if date_from is not None:
        where.append("p.first_version_date >= %(date_from)s")
        params["date_from"] = date_from
    if date_to is not None:
        where.append("p.first_version_date <= %(date_to)s")
        params["date_to"] = date_to
    # "C": the order of equal ranks and dates must not depend on the database's collation.
    order += ["p.first_version_date DESC", 'p.arxiv_id COLLATE "C"']
    sql = (
        "SELECT p.arxiv_id, p.title, p.first_version_date FROM paper p WHERE "
        + " AND ".join(where)
        + " ORDER BY "
        + ", ".join(order)
        + " LIMIT %(limit)s"
    )
    return sql, params


def search_papers(
    connection: psycopg.Connection,
    *,
    query: str | None,
    entity: str | None,
    category: str = "cs.CL",
    date_from: date | None,
    date_to: date | None,
    limit: int = 20,
    as_of: date | None = None,
) -> SearchResult:
    """Papers of a slice by text and entity. Raises BadRequest for a request that cannot be
    meant: a limit outside 1..100, or a date range the wrong way round."""
    if not 1 <= limit <= 100:
        raise BadRequest(f"limit must be from 1 to 100, got {limit}")
    if date_from and date_to and date_from > date_to:
        raise BadRequest(f"date_from {date_from} is after date_to {date_to}")
    header = response_header(connection, as_of=as_of)

    def empty(message: str) -> SearchResult:
        return SearchResult(header=header, papers=[], message=message)

    if category not in CATEGORIES:
        collected = ", ".join(CATEGORIES)
        return empty(f"Slice {category} is not collected. Collected slices: {collected}.")
    query = (query or "").strip() or None
    if query is not None and _is_empty_text_query(connection, query):
        return empty(f"The text {query!r} has no searchable words (only stop words or symbols).")
    canonical = None
    if entity is not None and entity.strip():
        run = current_run(connection)
        if run is None:
            return empty("No extraction run is current yet, so entities cannot be searched.")
        canonical = _canonical_name(connection, entity)
        if not _is_mentioned(connection, run, canonical):
            return empty(f"The current run has no mention of {entity!r}.")
    sql, params = build_query(
        query=query,
        entity=canonical,
        category=category,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        run=header.extraction_run,
    )
    papers = [
        PaperHit(arxiv_id=arxiv_id, title=title, date=day, url=f"https://arxiv.org/abs/{arxiv_id}")
        for arxiv_id, title, day in connection.execute(sql, params).fetchall()
    ]
    return SearchResult(
        header=header,
        papers=papers,
        message=None if papers else "No paper of the slice matches all the filters.",
    )


def _is_empty_text_query(connection: psycopg.Connection, query: str) -> bool:
    (nodes,) = connection.execute(f"SELECT numnode({_TEXT_QUERY})", {"query": query}).fetchone()
    return nodes == 0


def _canonical_name(connection: psycopg.Connection, entity: str) -> str:
    """The name a spelling is counted under: its normal form, then the dictionary (ADR 0002),
    as in the paper_entity view."""
    form = normal_form(entity)
    row = connection.execute(
        "SELECT canonical_name FROM synonym WHERE normal_form = %s", (form,)
    ).fetchone()
    return row[0] if row else form


def _is_mentioned(connection: psycopg.Connection, run: int, canonical: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM paper_entity WHERE run_id = %s AND canonical_name = %s LIMIT 1",
            (run, canonical),
        ).fetchone()
        is not None
    )
