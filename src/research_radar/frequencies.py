"""The most frequent entities of an extraction run, written for the agent to read.

The table's last three columns are filled in by the agent of the session that reads the
file; nothing here calls a model.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

# The papers of the slice that the run finished: the share's denominator.
_COMPLETED = """
    SELECT pe.arxiv_id
    FROM paper_extraction pe
    JOIN paper_category pc ON pc.arxiv_id = pe.arxiv_id AND pc.category = %(category)s
    WHERE pe.run_id = %(run_id)s
"""

_TOP = f"""
WITH completed AS ({_COMPLETED}),
counted AS (
    SELECT e.canonical_name, e.entity_type, count(*) AS papers
    FROM paper_entity e
    JOIN completed c USING (arxiv_id)
    WHERE e.run_id = %(run_id)s
    GROUP BY e.canonical_name, e.entity_type
)
SELECT canonical_name, entity_type, papers,
       papers::float8 / (SELECT count(*) FROM completed) AS share
FROM counted
ORDER BY share DESC, canonical_name COLLATE "C", entity_type
LIMIT %(limit)s
"""

# The view collapses spellings, so they are read from the mentions and put under the
# canonical name the same way the view does it.
_SPELLINGS = f"""
WITH completed AS ({_COMPLETED})
SELECT coalesce(s.canonical_name, m.normal_form) AS canonical_name, m.entity_type,
       m.spelling, count(*) AS uses
FROM mention m
JOIN completed c USING (arxiv_id)
LEFT JOIN synonym s USING (normal_form)
WHERE m.run_id = %(run_id)s
GROUP BY 1, 2, 3
"""

_EXAMPLES = f"""
WITH completed AS ({_COMPLETED}),
ranked AS (
    SELECT e.canonical_name, e.entity_type, e.arxiv_id,
           row_number() OVER (
               PARTITION BY e.canonical_name, e.entity_type ORDER BY e.arxiv_id COLLATE "C"
           ) AS n
    FROM paper_entity e
    JOIN completed c USING (arxiv_id)
    WHERE e.run_id = %(run_id)s
)
SELECT r.canonical_name, r.entity_type, p.arxiv_id, p.title, p.abstract
FROM ranked r
JOIN paper p USING (arxiv_id)
WHERE r.n <= %(examples)s
ORDER BY r.canonical_name COLLATE "C", r.entity_type, r.n
"""

_COMPLETED_COUNT = f"SELECT count(*) FROM ({_COMPLETED}) AS completed"


@dataclass(frozen=True)
class Example:
    arxiv_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class EntityFrequency:
    canonical_name: str
    entity_type: str
    papers: int
    share: float
    spellings: list[str]
    examples: list[Example]


def top_entities(
    connection: psycopg.Connection,
    run_id: int,
    *,
    category: str = "cs.CL",
    limit: int = 40,
    examples: int = 3,
) -> list[EntityFrequency]:
    parameters = {"run_id": run_id, "category": category, "limit": limit, "examples": examples}
    top = connection.execute(_TOP, parameters).fetchall()
    wanted = {(name, entity_type) for name, entity_type, _, _ in top}

    uses: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for name, entity_type, spelling, count in connection.execute(_SPELLINGS, parameters):
        if (name, entity_type) in wanted:
            uses.setdefault((name, entity_type), []).append((count, spelling))

    found: dict[tuple[str, str], list[Example]] = {}
    for name, entity_type, arxiv_id, title, abstract in connection.execute(_EXAMPLES, parameters):
        if (name, entity_type) in wanted:
            found.setdefault((name, entity_type), []).append(Example(arxiv_id, title, abstract))

    return [
        EntityFrequency(
            canonical_name=name,
            entity_type=entity_type,
            papers=papers,
            share=share,
            # Ties are broken in Python so the order does not depend on the collation.
            spellings=[
                s for _, s in sorted(uses[(name, entity_type)], key=lambda u: (-u[0], u[1]))
            ],
            examples=found.get((name, entity_type), []),
        )
        for name, entity_type, papers, share in top
    ]


def completed_papers(connection: psycopg.Connection, run_id: int, category: str) -> int:
    parameters = {"run_id": run_id, "category": category}
    return connection.execute(_COMPLETED_COUNT, parameters).fetchone()[0]


def _cell(text: str) -> str:
    """A table cell: a ``|`` or a line break would split the row."""
    return " ".join(text.replace("|", "\\|").split())


def write_table(
    rows: Sequence[EntityFrequency],
    path: Path,
    *,
    run_id: int,
    category: str,
    completed: int | None = None,
) -> None:
    lines = [f"# Частоты сущностей: прогон {run_id}, срез {category}", ""]
    if completed is not None:
        lines += [f"Статей с отметкой завершения в прогоне: {completed}.", ""]
    if not rows:
        lines += ["В прогоне нет статей с отметкой завершения в этом срезе: таблица пуста.", ""]
    lines += [
        "| # | имя | тип | статьи | доля | написания | фоновое | узкие соседи | рекомендация |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for number, row in enumerate(rows, 1):
        lines.append(
            f"| {number} | {_cell(row.canonical_name)} | {row.entity_type} | {row.papers} "
            f"| {row.share:.2f} | {_cell(', '.join(row.spellings))} |  |  |  |"
        )
    for number, row in enumerate(rows, 1):
        lines += ["", f"## {number}. {_cell(row.canonical_name)} ({row.entity_type})"]
        for example in row.examples:
            lines += [
                "",
                f"### {example.arxiv_id}: {_cell(example.title)}",
                "",
                *(f"> {line}" for line in example.abstract.splitlines() or [""]),
            ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
