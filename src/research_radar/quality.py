"""Extraction quality: a run compared with the gold set, and the log of the comparisons.

A match is the same paper, the same canonical name and the same type (ADR 0005); the
synonym dictionary applies to both sides (ADR 0002).
"""

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import psycopg
from pydantic import BaseModel, ValidationError

from research_radar import normalize
from research_radar.extraction import GIVEN_UP_SQL, require_idle
from research_radar.mentions import EntityType, Mention
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS
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
    # Gold papers the run gave up on: counted as an empty prediction (ADR 0005, #65).
    given_up_papers: list[str]


def read_gold(path: Path) -> list[GoldPaper]:
    """One paper per line. Raises ValueError naming the line of the first bad one."""
    return parse_gold(path.read_bytes(), path.name)


def parse_gold(raw: bytes, name: str) -> list[GoldPaper]:
    """The same of bytes already read, so a caller can hash exactly what was parsed."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name}: not UTF-8 text") from error
    papers: list[GoldPaper] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            paper = GoldPaper.model_validate_json(line)
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()
            )
            raise ValueError(f"{name}: line {number}: {problems}") from error
        if paper.arxiv_id in seen:
            raise ValueError(f"{name}: line {number}: {paper.arxiv_id} appears twice")
        seen.add(paper.arxiv_id)
        papers.append(paper)
    return papers


def rules_sha256() -> str:
    """The version of the normal form rules: the hash of the source of the normalize module."""
    return hashlib.sha256(inspect.getsource(normalize).encode()).hexdigest()


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
    previous_level = connection.isolation_level
    # One snapshot: the dictionary in the figures and the hash in the log are of one version.
    connection.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    try:
        with connection.transaction():
            return _measure(connection, run_id, gold, ids, gold_sha256, label)
    finally:
        connection.isolation_level = previous_level


def _measure(
    connection: psycopg.Connection,
    run_id: int,
    gold: list[GoldPaper],
    ids: list[str],
    gold_sha256: str,
    label: str,
) -> Measurement:
    # The loader takes this lock too, so a load in progress finishes before the read.
    connection.execute("SELECT pg_advisory_xact_lock(hashtext('synonym_load'))")
    # The share lock keeps the run from being deleted before the insert; without it that
    # would be a foreign key error instead of this message.
    if not connection.execute(
        "SELECT 1 FROM extraction_run WHERE id = %s FOR SHARE", (run_id,)
    ).fetchone():
        raise ValueError(f"no extraction run {run_id}")
    known = {
        r[0]
        for r in connection.execute("SELECT arxiv_id FROM paper WHERE arxiv_id = ANY(%s)", (ids,))
    }
    if missing := sorted(set(ids) - known):
        raise ValueError(f"gold papers not in the database: {', '.join(missing)}")
    # A paper whose abstract changed after the run read it is not finished for that run:
    # the mentions belong to the old text.
    current = {
        r[0]
        for r in connection.execute(
            "SELECT p.arxiv_id FROM paper p JOIN paper_extraction e ON e.arxiv_id = p.arxiv_id"
            " AND e.abstract_hash = p.abstract_hash"
            " WHERE e.run_id = %s AND p.arxiv_id = ANY(%s)",
            (run_id, ids),
        )
    }
    # A paper the run gave up on (the rule of the run itself) has no mark and never will: it
    # counts as an empty prediction, so the failure shows in the recall instead of blocking.
    given_up = {
        r[0]
        for r in connection.execute(
            GIVEN_UP_SQL, {"run": run_id, "attempts": EXTRACTION_MAX_ATTEMPTS}
        )
    } & (set(ids) - current)
    if unfinished := sorted(set(ids) - current - given_up):
        raise ValueError(
            f"run {run_id} has no completion mark for the current abstract of gold papers: "
            + ", ".join(unfinished)
        )
    synonyms = dict(connection.execute("SELECT normal_form, canonical_name FROM synonym"))
    if stale := sorted(key for key in synonyms if key != normalize.normal_form(key)):
        raise ValueError(
            "the synonym table has keys that are not normal forms by today's rules "
            f"({', '.join(map(repr, stale))}); run radar load-synonyms"
        )
    # The extraction side comes from the spellings, not from the stored normal form: the
    # rules of the normal form changed since (#48, #59), and both sides must follow today's.
    stored = connection.execute(
        "SELECT arxiv_id, entity_type, spelling FROM mention"
        " WHERE run_id = %s AND arxiv_id = ANY(%s)",
        (run_id, ids),
    ).fetchall()
    predicted = _triples([row for row in stored if row[0] not in given_up], synonyms)
    wanted = _triples(
        [(p.arxiv_id, m.entity_type, m.spelling) for p in gold for m in p.mentions], synonyms
    )
    dictionary = current_dictionary(connection)
    result = Measurement(
        run_id=run_id,
        dictionary_sha256=dictionary.file_sha256 if dictionary else None,
        gold_sha256=gold_sha256,
        label=label,  # type: ignore[arg-type]  # checked against LABELS by the caller
        by_type={
            kind: _score(
                {t for t in predicted if t[2] == kind}, {t for t in wanted if t[2] == kind}
            )
            for kind in TYPES
        },
        overall_typed=_score(predicted, wanted),
        overall_untyped=_score({t[:2] for t in predicted}, {t[:2] for t in wanted}),
        given_up_papers=sorted(given_up),
    )
    connection.execute(
        "INSERT INTO quality_measurement (run_id, dictionary_sha256, gold_sha256, rules_sha256,"
        " label, metrics) VALUES (%s, %s, %s, %s, %s, %s)",
        (
            run_id,
            result.dictionary_sha256,
            gold_sha256,
            rules_sha256(),
            label,
            json.dumps(
                {
                    "by_type": {k: asdict(v) for k, v in result.by_type.items()},
                    "overall_typed": asdict(result.overall_typed),
                    "overall_untyped": asdict(result.overall_untyped),
                    "given_up_papers": {
                        "count": len(result.given_up_papers),
                        "ids": result.given_up_papers,
                    },
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
        form = normalize.normal_form(spelling)
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


def format_scores(
    by_type: dict[str, Score], overall_typed: Score, overall_untyped: Score
) -> list[str]:
    """The lines of one measurement; `measure` and `report` print the figures alike."""
    return [
        *(_score_line(kind, score) for kind, score in by_type.items()),
        _score_line("overall, typed", overall_typed),
        _score_line("overall, untyped", overall_untyped),
    ]


def _given_up_line(ids: list[str]) -> str:
    return f"  given up: {len(ids)} ({', '.join(ids)})" if ids else "  given up: 0"


def format_measurement(result: Measurement) -> str:
    lines = format_scores(result.by_type, result.overall_typed, result.overall_untyped)
    return "\n".join([*lines, _given_up_line(result.given_up_papers), CAVEAT])


def _score_from(record: object) -> Score:
    # Rows written by an earlier version of the code may lack fields.
    data = record if isinstance(record, dict) else {}
    return Score(
        precision=data.get("precision"),
        recall=data.get("recall"),
        true_positives=data.get("true_positives", 0),
        predicted=data.get("predicted", 0),
        gold=data.get("gold", 0),
    )


_LOG_COLUMNS = "measured_at, run_id, dictionary_sha256, rules_sha256, gold_sha256, metrics"


def _block(label: str, row) -> list[str]:
    measured_at, run_id, dictionary, rules, gold_hash, metrics = row
    metrics = metrics if isinstance(metrics, dict) else {}
    by_type = {kind: _score_from(v) for kind, v in (metrics.get("by_type") or {}).items()}
    typed = _score_from(metrics.get("overall_typed"))
    untyped = _score_from(metrics.get("overall_untyped"))
    given_up = metrics.get("given_up_papers")  # rows of earlier versions have none
    ids = [str(i) for i in given_up.get("ids", [])] if isinstance(given_up, dict) else []
    return [
        f"{label}: {measured_at:%Y-%m-%d}, run {run_id}, dictionary {(dictionary or 'none')[:12]}, "
        f"rules {(rules or 'none')[:12]}, gold {gold_hash[:12]}",
        *format_scores(by_type, typed, untyped),
        _given_up_line(ids),
    ]


def report(connection: psycopg.Connection) -> str:
    """The latest 'after' and the latest 'before' measured on the same basis as it (run,
    dictionary and rules), each figure with the number of mentions behind it."""
    after = connection.execute(
        f"SELECT {_LOG_COLUMNS} FROM quality_measurement WHERE label = 'after'"
        " ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if after is None:
        before = connection.execute(
            f"SELECT {_LOG_COLUMNS} FROM quality_measurement WHERE label = 'before'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        lines = _block("before", before) if before else ["before: no measurements"]
        lines.append("after: no measurements")
        return "\n".join([*lines, CAVEAT])
    before = connection.execute(
        f"SELECT {_LOG_COLUMNS} FROM quality_measurement WHERE label = 'before'"
        " AND run_id = %s AND dictionary_sha256 IS NOT DISTINCT FROM %s AND rules_sha256 = %s"
        " ORDER BY id DESC LIMIT 1",
        (after[1], after[2], after[3]),
    ).fetchone()
    lines: list[str] = []
    if before is None:
        before = connection.execute(
            f"SELECT {_LOG_COLUMNS} FROM quality_measurement WHERE label = 'before'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if before:
            lines.append(
                "warning: no 'before' measurement has the run, dictionary and rules of the "
                "'after'; the two below are on a different basis and are not comparable"
            )
    lines += _block("before", before) if before else ["before: no measurements"]
    lines += _block("after", after)
    return "\n".join([*lines, CAVEAT])
