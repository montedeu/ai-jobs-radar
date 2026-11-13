"""Extraction quality: a run compared with the gold set, and the log of the comparisons.

A match is the same paper, the same canonical name and the same type (ADR 0005); the
synonym dictionary applies to both sides (ADR 0002).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import psycopg
from pydantic import BaseModel, ValidationError

from research_radar.extraction import require_idle
from research_radar.mentions import EntityType, Mention
from research_radar.normalize import normal_form
from research_radar.synonyms import current_dictionary

LABELS = ("before", "after")
TYPES: tuple[str, ...] = get_args(EntityType)
# What the service says about every figure of quality (CLAUDE.md, "Что сервис оговаривает").
CAVEAT = "The figures are for cs.CL and for the rules as read by one annotator."


class GoldPaper(BaseModel):
    arxiv_id: str
    mentions: list[Mention]  # the expansion is not compared


@dataclass(frozen=True)
class Score:
    precision: float | None  # None when nothing was predicted
    recall: float | None  # None when the gold has no mention
    true_positives: int
    predicted: int  # unique (paper, name, type) triples of the extraction
    gold: int  # the same of the gold set


@dataclass(frozen=True)
class Measurement:
    run_id: int
    dictionary_sha256: str | None
    gold_sha256: str
    label: Literal["before", "after"]
    by_type: dict[str, Score]
    overall_typed: Score
    overall_untyped: Score


def read_gold(path: Path) -> list[GoldPaper]:
    """One paper per line. Raises ValueError naming the line of the first bad one."""
    papers: list[GoldPaper] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                paper = GoldPaper.model_validate_json(line)
            except ValidationError as error:
                problems = "; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
                )
                raise ValueError(f"{path.name}: line {number}: {problems}") from error
            if paper.arxiv_id in seen:
                raise ValueError(f"{path.name}: line {number}: {paper.arxiv_id} appears twice")
            seen.add(paper.arxiv_id)
            papers.append(paper)
    return papers


def _score(predicted: set, gold: set) -> Score:
    hits = len(predicted & gold)
    return Score(
        precision=hits / len(predicted) if predicted else None,
        recall=hits / len(gold) if gold else None,
        true_positives=hits,
        predicted=len(predicted),
        gold=len(gold),
    )


def measure(
    connection: psycopg.Connection,
    run_id: int,
    gold: list[GoldPaper],
    *,
    gold_sha256: str,
    label: str,
) -> Measurement:
    """Compares the run with the gold set and writes a row to the log."""
    require_idle(connection)
    if label not in LABELS:
        raise ValueError(f"label must be one of {', '.join(LABELS)}, not {label!r}")
    if not gold:
        raise ValueError("the gold set is empty")
    ids = [paper.arxiv_id for paper in gold]
    with connection.transaction():
        # Without the lock a run deleted meanwhile would fail the insert with a foreign key
        # error instead of this message.
        if not connection.execute(
            "SELECT 1 FROM extraction_run WHERE id = %s FOR SHARE", (run_id,)
        ).fetchone():
            raise ValueError(f"no extraction run {run_id}")
        known = {
            r[0]
            for r in connection.execute(
                "SELECT arxiv_id FROM paper WHERE arxiv_id = ANY(%s)", (ids,)
            )
        }
        if missing := sorted(set(ids) - known):
            raise ValueError(f"gold papers not in the database: {', '.join(missing)}")
        done = {
            r[0]
            for r in connection.execute(
                "SELECT arxiv_id FROM paper_extraction WHERE run_id = %s AND arxiv_id = ANY(%s)",
                (run_id, ids),
            )
        }
        if unfinished := sorted(set(ids) - done):
            raise ValueError(
                f"run {run_id} has no completion mark for gold papers: {', '.join(unfinished)}"
            )
        synonyms = dict(connection.execute("SELECT normal_form, canonical_name FROM synonym"))
        # The extraction side comes from the spellings, not from the stored normal form: the
        # rules of the normal form changed since (#48, #59), and both sides must follow today's.
        stored = connection.execute(
            "SELECT arxiv_id, entity_type, spelling FROM mention"
            " WHERE run_id = %s AND arxiv_id = ANY(%s)",
            (run_id, ids),
        ).fetchall()
        predicted = _triples(stored, synonyms)
        wanted = _triples(
            [(p.arxiv_id, m.entity_type, m.spelling) for p in gold for m in p.mentions], synonyms
        )
        dictionary = current_dictionary(connection)
        result = Measurement(
            run_id=run_id,
            dictionary_sha256=dictionary.file_sha256 if dictionary else None,
            gold_sha256=gold_sha256,
            label=label,  # type: ignore[arg-type]  # checked against LABELS above
            by_type={
                kind: _score(
                    {t for t in predicted if t[2] == kind}, {t for t in wanted if t[2] == kind}
                )
                for kind in TYPES
            },
            overall_typed=_score(predicted, wanted),
            overall_untyped=_score({t[:2] for t in predicted}, {t[:2] for t in wanted}),
        )
        connection.execute(
            "INSERT INTO quality_measurement (run_id, dictionary_sha256, gold_sha256, label,"
            " metrics) VALUES (%s, %s, %s, %s, %s)",
            (
                run_id,
                result.dictionary_sha256,
                gold_sha256,
                label,
                json.dumps(
                    {
                        "by_type": {k: asdict(v) for k, v in result.by_type.items()},
                        "overall_typed": asdict(result.overall_typed),
                        "overall_untyped": asdict(result.overall_untyped),
                    }
                ),
            ),
        )
    return result


def _triples(rows, synonyms: dict[str, str]) -> set[tuple[str, str, str]]:
    """(paper, canonical name, type), by the same rule as the paper_entity view; a spelling
    of nothing but punctuation names nothing and is dropped, as when mentions are stored."""
    triples = set()
    for arxiv_id, entity_type, spelling in rows:
        form = normal_form(spelling)
        if form:
            triples.add((arxiv_id, synonyms.get(form, form), entity_type))
    return triples


def _figure(name: str, value: float | None, hits: int, total: int) -> str:
    shown = "n/a" if value is None else f"{value:.2f}"
    return f"{name} {shown} ({hits}/{total})"


def _score_line(name: str, score: Score) -> str:
    return (
        f"  {name}: "
        + _figure("precision", score.precision, score.true_positives, score.predicted)
        + ", "
        + _figure("recall", score.recall, score.true_positives, score.gold)
    )


def report(connection: psycopg.Connection) -> str:
    """The latest measurement of each label, with the number of mentions behind each figure."""
    lines: list[str] = []
    for label in LABELS:
        row = connection.execute(
            "SELECT id, measured_at, run_id, dictionary_sha256, gold_sha256, metrics"
            " FROM quality_measurement WHERE label = %s ORDER BY id DESC LIMIT 1",
            (label,),
        ).fetchone()
        if row is None:
            lines.append(f"{label}: no measurements")
            continue
        _, measured_at, run_id, dictionary, gold_hash, metrics = row
        lines.append(
            f"{label}: {measured_at:%Y-%m-%d}, run {run_id}, "
            f"dictionary {(dictionary or 'none')[:12]}, gold {gold_hash[:12]}"
        )
        for kind in TYPES:
            lines.append(_score_line(kind, Score(**metrics["by_type"][kind])))
        lines.append(_score_line("overall, typed", Score(**metrics["overall_typed"])))
        lines.append(_score_line("overall, untyped", Score(**metrics["overall_untyped"])))
    lines.append(CAVEAT)
    return "\n".join(lines)
