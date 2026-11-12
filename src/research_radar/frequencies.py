"""The share of papers of each entity in an extraction run, written for the agent to read.

The table's last three columns are filled in by the agent of the session that reads the
file; nothing here calls a model.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import psycopg

DEFAULT_CATEGORY = "cs.CL"
DEFAULT_LIMIT = 40
DEFAULT_EXAMPLES = 3

# ``completed`` is the papers of the slice that the run finished: the share's denominator.
# ``top`` is the rows that get spellings and examples, so those are never read for the
# whole run.
_TOP = """
WITH completed AS (
    SELECT pe.arxiv_id
    FROM paper_extraction pe
    JOIN paper_category pc ON pc.arxiv_id = pe.arxiv_id AND pc.category = %(category)s
    WHERE pe.run_id = %(run_id)s
),
counted AS (
    SELECT e.canonical_name, e.entity_type, count(*) AS papers
    FROM paper_entity e
    JOIN completed c USING (arxiv_id)
    WHERE e.run_id = %(run_id)s
    GROUP BY e.canonical_name, e.entity_type
),
top AS (
    SELECT canonical_name, entity_type, papers,
           papers::float8 / (SELECT count(*) FROM completed) AS share
    FROM counted
    ORDER BY share DESC, canonical_name COLLATE "C", entity_type
    LIMIT %(limit)s
)
"""

_COMPLETED_COUNT = """
SELECT count(*)
FROM paper_extraction pe
JOIN paper_category pc ON pc.arxiv_id = pe.arxiv_id AND pc.category = %(category)s
WHERE pe.run_id = %(run_id)s
"""

_ROWS = (
    _TOP
    + """
SELECT canonical_name, entity_type, papers, share, (SELECT count(*) FROM completed)
FROM top
ORDER BY share DESC, canonical_name COLLATE "C", entity_type
"""
)

# The view collapses spellings, so they are read from the mentions and put under the
# canonical name the same way the view does it.
_SPELLINGS = (
    _TOP
    + """
SELECT t.canonical_name, t.entity_type, m.spelling, count(*) AS uses
FROM mention m
JOIN completed c USING (arxiv_id)
LEFT JOIN synonym s USING (normal_form)
JOIN top t ON t.canonical_name = coalesce(s.canonical_name, m.normal_form)
          AND t.entity_type = m.entity_type
WHERE m.run_id = %(run_id)s
GROUP BY 1, 2, 3
"""
)

_EXAMPLES = (
    _TOP
    + """
, ranked AS (
    SELECT e.canonical_name, e.entity_type, e.arxiv_id,
           row_number() OVER (
               PARTITION BY e.canonical_name, e.entity_type ORDER BY e.arxiv_id COLLATE "C"
           ) AS n
    FROM paper_entity e
    JOIN top t USING (canonical_name, entity_type)
    JOIN completed c USING (arxiv_id)
    WHERE e.run_id = %(run_id)s
)
SELECT r.canonical_name, r.entity_type, p.arxiv_id, p.title, p.abstract,
       x.abstract_hash <> p.abstract_hash AS changed
FROM ranked r
JOIN paper p USING (arxiv_id)
JOIN paper_extraction x ON x.run_id = %(run_id)s AND x.arxiv_id = r.arxiv_id
WHERE r.n <= %(examples)s
ORDER BY r.canonical_name COLLATE "C", r.entity_type, r.n
"""
)

_SNAPSHOT = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"


@dataclass(frozen=True)
class Example:
    arxiv_id: str
    title: str
    abstract: str
    # The abstract was edited after the run read it, so the mentions may not match the text.
    changed: bool


@dataclass(frozen=True)
class EntityFrequency:
    canonical_name: str
    entity_type: str
    papers: int
    share: float
    spellings: list[str]
    examples: list[Example]


@dataclass(frozen=True)
class FrequencyTable:
    run_id: int
    category: str
    completed: int  # papers of the slice with a completion mark: the shares' denominator
    rows: list[EntityFrequency]


def top_entities(
    connection: psycopg.Connection,
    run_id: int,
    *,
    category: str = DEFAULT_CATEGORY,
    limit: int = DEFAULT_LIMIT,
    examples: int = DEFAULT_EXAMPLES,
) -> FrequencyTable:
    """Needs a connection with no transaction open: the queries share one snapshot."""
    parameters = {"run_id": run_id, "category": category, "limit": limit, "examples": examples}
    # Without one snapshot a collection landing between the queries leaves a row with no
    # spellings, or a spelling with no row.
    with connection.transaction():
        connection.execute(_SNAPSHOT)
        top = connection.execute(_ROWS, parameters).fetchall()
        uses: dict[tuple[str, str], list[tuple[int, str]]] = {}
        found: dict[tuple[str, str], list[Example]] = {}
        if top:
            completed = top[0][4]
            for name, entity_type, spelling, count in connection.execute(_SPELLINGS, parameters):
                uses.setdefault((name, entity_type), []).append((count, spelling))
            for name, entity_type, *example in connection.execute(_EXAMPLES, parameters):
                found.setdefault((name, entity_type), []).append(Example(*example))
        else:
            completed = connection.execute(_COMPLETED_COUNT, parameters).fetchone()[0]

    rows = [
        EntityFrequency(
            canonical_name=name,
            entity_type=entity_type,
            papers=papers,
            share=share,
            # Ties are broken in Python so the order does not depend on the collation.
            spellings=[spelling for _, spelling in sorted(uses[(name, entity_type)], key=_by_use)],
            examples=found[(name, entity_type)],
        )
        for name, entity_type, papers, share, _ in top
    ]
    return FrequencyTable(run_id=run_id, category=category, completed=completed, rows=rows)


def _by_use(use: tuple[int, str]) -> tuple[int, str]:
    return -use[0], use[1]


def _escape(text: str) -> str:
    """Text in a table cell or a heading: nothing in it may be read as Markdown."""
    text = " ".join(text.split())
    return "".join("\\" + c if c in "\\`*_[]<>#|~&" else c for c in text)


def _fenced(text: str) -> list[str]:
    # A fence longer than any run of backticks inside, so the text cannot close it early.
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}text", *text.splitlines(), fence]


def write_table(table: FrequencyTable, path: Path) -> None:
    lines = [
        f"# Доли сущностей: прогон {table.run_id}, срез {table.category}",
        "",
        f"Статей с отметкой завершения в прогоне: {table.completed}.",
        "",
    ]
    if table.completed == 0:
        lines += ["В прогоне нет статей с отметкой завершения в этом срезе.", ""]
    elif not table.rows:
        lines += ["Сущностей нет: ни в одной из этих статей ничего не найдено.", ""]
    lines += [
        "| # | имя | тип | статьи | доля | написания | фоновое | узкие соседи | рекомендация |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for number, row in enumerate(table.rows, 1):
        spellings = " · ".join(_escape(s) for s in row.spellings)
        lines.append(
            f"| {number} | {_escape(row.canonical_name)} | {row.entity_type} | {row.papers} "
            f"| {row.share:.2f} | {spellings} |  |  |  |"
        )
    for number, row in enumerate(table.rows, 1):
        lines += ["", f"## {number}. {_escape(row.canonical_name)} ({row.entity_type})"]
        for example in row.examples:
            lines += ["", f"### {example.arxiv_id}", ""]
            if example.changed:
                lines += ["Аннотация изменилась после извлечения.", ""]
            lines += _fenced(f"{example.title}\n\n{example.abstract}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
