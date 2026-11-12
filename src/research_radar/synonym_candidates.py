"""Candidates for the synonym dictionary: "Long Form (ABBR)" pairs found in abstracts.

A candidate takes effect only when the developer moves the row into ``synonyms.csv`` (ADR 0002);
nothing here writes the dictionary or the ``synonym`` table. Grouping is by ``normal_form()``,
the key the dictionary itself uses, so "LoRA" and "LORA" are one abbreviation.
"""

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import psycopg

from research_radar.normalize import normal_form

_MAX_EXPANSION_WORDS = 8
_PARENTHESIS = re.compile(r"\(([^()]*)\)")
_ABBREVIATION = re.compile(r"[A-Za-z0-9-]{2,8}")
_EDGE_PUNCTUATION = "\"'“”‘’[]{}"
_CLAUSE_END = (".", ",", ";", ":", "!", "?")
_COLUMNS = ["normal_form", "canonical_name", "papers"]


@dataclass(frozen=True)
class Pair:
    expansion: str
    abbreviation: str


@dataclass(frozen=True)
class Candidates:
    # normal form of the abbreviation -> (expansion, number of papers)
    pairs: dict[str, tuple[str, int]]
    # abbreviation with two or more expansions -> {expansion: number of papers}
    conflicts: dict[str, dict[str, int]]


def find_pairs(abstract: str) -> list[Pair]:
    """Pairs written as "Long Form (ABBR)" or "ABBR (Long Form)", each once, in text order."""
    pairs: list[Pair] = []
    text = abstract
    # Innermost parentheses first; cutting them out lets an enclosing pair be read next round.
    while matches := list(_PARENTHESIS.finditer(text)):
        for match in matches:
            pair = _pair_at(text, match)
            if pair and pair not in pairs:
                pairs.append(pair)
        text = _PARENTHESIS.sub("", text)
    return pairs


def _pair_at(text: str, match: re.Match[str]) -> Pair | None:
    inside = match.group(1).strip(" " + _EDGE_PUNCTUATION)
    before = text[: match.start()].rstrip()
    if _is_abbreviation(inside) and (expansion := _expansion_before(before, inside)):
        return Pair(expansion, inside)
    words = inside.split()
    if 0 < len(words) <= _MAX_EXPANSION_WORDS and before:
        abbreviation = before.split()[-1].strip(_EDGE_PUNCTUATION)
        if _is_abbreviation(abbreviation) and _spells(abbreviation, inside):
            return Pair(inside, abbreviation)
    return None


def _is_abbreviation(token: str) -> bool:
    return bool(_ABBREVIATION.fullmatch(token)) and sum(c.isupper() for c in token) >= 2


def _expansion_before(before: str, abbreviation: str) -> str | None:
    """The shortest run of words ending at the parenthesis that the abbreviation spells."""
    words = before.split()
    for count in range(1, min(_MAX_EXPANSION_WORDS, len(words)) + 1):
        run = words[-count:]
        # A comma or full stop ends the clause: words before it are not part of the name.
        if any(word.endswith(_CLAUSE_END) for word in run[:-1]):
            return None
        candidate = " ".join(run).lstrip(_EDGE_PUNCTUATION)
        if candidate and _spells(abbreviation, candidate):
            return candidate
    return None


def _spells(abbreviation: str, expansion: str) -> bool:
    """The first letter starts the expansion; the others follow in order, anywhere after it."""
    letters = [c.lower() for c in abbreviation if c.isalpha()]
    text = expansion.lower()
    if len(letters) < 2 or not text.startswith(letters[0]):
        return False
    position = 1
    for letter in letters[1:]:
        position = text.find(letter, position) + 1
        if position == 0:
            return False
    return True


def collect_candidates(connection: psycopg.Connection) -> Candidates:
    """Pairs and conflicts over every stored abstract, minus abbreviations the dictionary has."""
    known = {row[0] for row in connection.execute("SELECT normal_form FROM synonym")}
    # abbreviation normal form -> expansion normal form -> spelling -> papers
    spellings: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for (abstract,) in connection.execute("SELECT abstract FROM paper"):
        for pair in find_pairs(abstract):
            key = normal_form(pair.abbreviation)
            if key not in known:
                spellings[key][normal_form(pair.expansion)][pair.expansion] += 1

    pairs: dict[str, tuple[str, int]] = {}
    conflicts: dict[str, dict[str, int]] = {}
    for key, expansions in spellings.items():
        counted = {_most_common(spelled): _papers(spelled) for spelled in expansions.values()}
        if len(counted) == 1:
            ((expansion, papers),) = counted.items()
            pairs[key] = (expansion, papers)
        else:
            conflicts[key] = counted
    return Candidates(
        pairs=dict(sorted(pairs.items(), key=lambda item: (-item[1][1], item[0]))),
        conflicts=dict(sorted(conflicts.items())),
    )


def _papers(spelled: Counter[str]) -> int:
    # find_pairs gives each pair once per paper, so spelling counts are paper counts.
    return sum(spelled.values())


def _most_common(spelled: Counter[str]) -> str:
    return min(spelled, key=lambda spelling: (-spelled[spelling], spelling))


def write_candidates_csv(candidates: Candidates, path: Path) -> None:
    """Pairs to ``path``; conflicts, one row per expansion, to the ``.conflicts.csv`` beside it."""
    pair_rows = [(key, name, papers) for key, (name, papers) in candidates.pairs.items()]
    conflict_rows = [
        (key, name, papers)
        for key, expansions in candidates.conflicts.items()
        for name, papers in sorted(expansions.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write(path, pair_rows)
    _write(path.with_suffix(".conflicts.csv"), conflict_rows)


def _write(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)
