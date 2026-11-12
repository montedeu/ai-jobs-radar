"""Stores Papers: one row per arXiv id, always holding the latest version seen."""

import hashlib
from typing import Literal

import psycopg
from psycopg.types.json import Jsonb

from research_radar.arxiv_api import Paper

Outcome = Literal["new", "abstract_changed", "updated", "unchanged"]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def abstract_hash(abstract: str) -> str:
    return text_hash(abstract)


def store(connection: psycopg.Connection, paper: Paper) -> Outcome:
    """Inserts or updates the Paper and says what changed.

    The date of the first version is written once and never updated. A changed abstract
    is what later sends the Paper back to extraction (docs/adr/0003); nothing is queued
    here, the changed hash is the signal.
    """
    new_hash = abstract_hash(paper.abstract)
    existing = connection.execute(
        "SELECT abstract_hash, version FROM paper WHERE arxiv_id = %s FOR UPDATE",
        (paper.arxiv_id,),
    ).fetchone()
    if existing and existing[1] >= paper.version:
        outcome: Outcome = "unchanged"
    elif existing:
        outcome = "abstract_changed" if existing[0] != new_hash else "updated"
    else:
        outcome = "new"
    if outcome == "unchanged":
        if existing[1] == paper.version:
            # arXiv adds a category to a paper without a new version. An older version,
            # served from a stale page, must not undo that.
            _replace_categories(connection, paper)
        return outcome
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%(arxiv_id)s, %(version)s, %(title)s, %(abstract)s, %(hash)s, %(authors)s,
                %(primary_category)s, %(first_version_date)s, %(arxiv_updated_at)s)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            version = excluded.version,
            title = excluded.title,
            abstract = excluded.abstract,
            abstract_hash = excluded.abstract_hash,
            authors = excluded.authors,
            primary_category = excluded.primary_category,
            arxiv_updated_at = excluded.arxiv_updated_at,
            updated_at = now()
        """,
        {
            **paper.model_dump(exclude={"authors", "categories"}),
            "hash": new_hash,
            "authors": Jsonb(paper.authors),
        },
    )
    _replace_categories(connection, paper)
    return outcome


def _replace_categories(connection: psycopg.Connection, paper: Paper) -> None:
    connection.execute(
        "DELETE FROM paper_category WHERE arxiv_id = %s AND category <> ALL(%s)",
        (paper.arxiv_id, paper.categories),
    )
    connection.execute(
        """
        INSERT INTO paper_category (arxiv_id, category)
        SELECT %s, unnest(%s::text[]) ON CONFLICT DO NOTHING
        """,
        (paper.arxiv_id, paper.categories),
    )
