"""Candidates for the synonym dictionary: "Long Form (ABBR)" pairs found in abstracts.

A candidate takes effect only when the developer moves the row into ``synonyms.csv`` (ADR 0002);
nothing here writes the dictionary or the ``synonym`` table. Grouping is by ``normal_form()``,
the key the dictionary itself uses, so "LoRA" and "LORA" are one abbreviation. The candidates
file has a ``papers`` column that ``read_synonyms`` rejects: the developer copies the first two
columns of the rows they accept.
"""

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import psycopg

from research_radar.normalize import normal_form

_MAX_EXPANSION_WORDS = 8
_PARENTHESIS = re.compile(r"\(([^()]*)\)")
_ABBREVIATION = re.compile(r"[A-Za-z-]{2,8}")
_EDGE_PUNCTUATION = "\"'“”‘’[]{}()"
_CLAUSE_END = (".", ",", ";", ":", "!", "?")
# What follows these words is a pointer or an example, not the long form of an abbreviation.
_NOT_AN_EXPANSION = re.compile(
    r"(?:see|e\.?g\.?|i\.?e\.?|cf\.?|like|such as|for example|including)(?![a-z])", re.I
)
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
    found: list[tuple[int, Pair]] = []
    text = abstract
    # Innermost parentheses first; blanking them (same length, so positions stay valid) lets an
    # enclosing pair be read in the next round.
    while matches := list(_PARENTHESIS.finditer(text)):
        for match in matches:
            if pair := _pair_at(text, match):
                found.append((match.start(), pair))
        text = _blank_groups(text, once=True)
    pairs: list[Pair] = []
    for _, pair in sorted(found, key=lambda item: item[0]):
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _blank_groups(text: str, once: bool = False) -> str:
    while _PARENTHESIS.search(text):
        text = _PARENTHESIS.sub(lambda group: " " * len(group.group()), text)
        if once:
            break
    return text


def _pair_at(text: str, match: re.Match[str]) -> Pair | None:
    # A citation or a second remark after ";" or "," is not part of the name.
    inside = re.split(r"[;,]", match.group(1))[0].strip(" " + _EDGE_PUNCTUATION)
    before = _blank_groups(text[: match.start()]).rstrip()
    pair = None
    if _is_abbreviation(inside) and (expansion := _expansion_before(before, inside)):
        pair = Pair(expansion, inside)
    elif 0 < len(inside.split()) <= _MAX_EXPANSION_WORDS and before:
        abbreviation = before.split()[-1].strip(_EDGE_PUNCTUATION)
        if _is_abbreviation(abbreviation) and _spell_score(abbreviation, inside) is not None:
            pair = Pair(inside, abbreviation)
    return pair if pair and _is_plausible(pair) else None


def _is_abbreviation(token: str) -> bool:
    return bool(_ABBREVIATION.fullmatch(token)) and sum(c.isupper() for c in token) >= 2


def _is_plausible(pair: Pair) -> bool:
    if any(c.isdigit() for c in pair.expansion) or _NOT_AN_EXPANSION.match(pair.expansion):
        return False
    # "BERT (BERT-base)" names a variant of the model, not its long form.
    abbreviation, expansion = normal_form(pair.abbreviation), normal_form(pair.expansion)
    return abbreviation not in expansion and expansion not in abbreviation


def _expansion_before(before: str, abbreviation: str) -> str | None:
    """The run of words before the parenthesis that the abbreviation spells best, then shortest."""
    words = before.split()
    best, best_score = None, -1
    for count in range(1, min(_MAX_EXPANSION_WORDS, len(words)) + 1):
        run = words[-count:]
        # A comma or full stop ends the clause: words before it are not part of the name.
        if any(word.endswith(_CLAUSE_END) for word in run[:-1]):
            break
        candidate = " ".join(run).lstrip(_EDGE_PUNCTUATION)
        score = _spell_score(abbreviation, candidate) if candidate else None
        if score is not None and score > best_score:
            best, best_score = candidate, score
    return best


def _spell_score(abbreviation: str, expansion: str) -> int | None:
    """How many letters of the abbreviation fall on word starts, or None if it does not spell it.

    The first letter starts the expansion, every capital starts a word (split at spaces and
    hyphens), lowercase letters may sit inside a word. All in the abbreviation's order.
    """
    letters = [c for c in abbreviation if c.isalpha()]
    text = expansion.lower()
    starts = [i == 0 or not expansion[i - 1].isalnum() for i in range(len(expansion))]
    if len(letters) < 2 or not text.startswith(letters[0].lower()):
        return None

    @cache
    def best(index: int, position: int) -> int | None:
        if index == len(letters):
            return 0
        letter = letters[index]
        top = None
        for at in range(position, len(text)):
            if text[at] != letter.lower() or (letter.isupper() and not starts[at]):
                continue
            if index == 0 and at != 0:
                break
            rest = best(index + 1, at + 1)
            if rest is not None and (top is None or rest + starts[at] > top):
                top = rest + starts[at]
        return top

    return best(0, 0)


def collect_candidates(connection: psycopg.Connection) -> Candidates:
    """Pairs and conflicts over every stored abstract, minus the pairs the dictionary covers.

    A pair is covered when both its normal forms are dictionary keys with one canonical name;
    an abbreviation the dictionary knows only in part still shows up.
    """
    known = dict(connection.execute("SELECT normal_form, canonical_name FROM synonym"))
    # abbreviation normal form -> expansion normal form -> spelling -> papers
    spellings: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    # Streamed: the corpus is every abstract collected, too many to hold at once.
    with connection.cursor() as cursor:
        for (abstract,) in cursor.stream("SELECT abstract FROM paper"):
            for pair in find_pairs(abstract):
                key, expansion_key = normal_form(pair.abbreviation), normal_form(pair.expansion)
                if key in known and expansion_key in known and known[key] == known[expansion_key]:
                    continue
                spellings[key][expansion_key][pair.expansion] += 1

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
    """Pairs to ``path``; conflicts, one row per expansion, to the ``.conflicts.csv`` beside it.

    A pair is two rows with one canonical name, the abbreviation's and the expansion's normal
    form: mentions of the expansion are matched by its own normal form, not the abbreviation's.
    """
    pair_rows: list[tuple[str, str, int]] = []
    named: dict[str, str] = {}  # expansion normal form -> canonical name already written for it
    for key, (name, papers) in candidates.pairs.items():
        expansion_key = normal_form(name)
        new = expansion_key not in named
        canonical = named.setdefault(expansion_key, name)
        pair_rows.append((key, canonical, papers))
        if new:
            pair_rows.append((expansion_key, canonical, papers))
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
