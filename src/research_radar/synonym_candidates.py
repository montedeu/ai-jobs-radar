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
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import psycopg

from research_radar.normalize import normal_form

_MAX_EXPANSION_WORDS = 8
_PARENTHESIS = re.compile(r"\(([^()]*)\)")
_BRACKETS = re.compile(r"\[[^\[\]]*\]")
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
    # abbreviation -> canonical name the dictionary already has for it or its expansion; a
    # candidate row takes that name instead of renaming the existing entity
    existing_names: dict[str, str] = field(default_factory=dict)


def find_pairs(abstract: str) -> list[Pair]:
    """Pairs written as "Long Form (ABBR)" or "ABBR (Long Form)", each once, in text order."""
    found: list[tuple[int, Pair]] = []
    # Citations in square brackets are blanked like the parentheses below: same length, so
    # positions stay valid, and no digit of a reference reaches the checks.
    text = _BRACKETS.sub(_blank, abstract)
    # Innermost parentheses first; blanking them lets an enclosing pair be read in the next round.
    while matches := list(_PARENTHESIS.finditer(text)):
        text = _PARENTHESIS.sub(_blank, text)
        for match in matches:
            if pair := _pair_at(text, match):
                found.append((match.start(), pair))
    pairs: list[Pair] = []
    for _, pair in sorted(found, key=lambda item: item[0]):
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def _blank(match: re.Match[str]) -> str:
    return " " * len(match.group())


def _pair_at(blanked: str, match: re.Match[str]) -> Pair | None:
    """``blanked`` is the text with this round's innermost groups already blanked."""
    # A citation or a second remark after ";" or "," is not part of the name.
    inside = re.split(r"[;,]", match.group(1))[0].strip(" " + _EDGE_PUNCTUATION)
    before = blanked[: match.start()].rstrip()
    pair = None
    if _is_abbreviation(inside) and (expansion := _expansion_before(before, inside)):
        pair = Pair(expansion, inside)
    elif 0 < len(inside.split()) <= _MAX_EXPANSION_WORDS and before:
        abbreviation = before.split()[-1].strip(_EDGE_PUNCTUATION)
        if _is_abbreviation(abbreviation) and _ends_in_last_word(abbreviation, inside):
            pair = Pair(inside, abbreviation)
    return pair if pair and _is_plausible(pair) else None


def _is_abbreviation(token: str) -> bool:
    return bool(_ABBREVIATION.fullmatch(token)) and sum(c.isupper() for c in token) >= 2


def _ends_in_last_word(abbreviation: str, expansion: str) -> bool:
    # "RL (reinforcement learning from human feedback)" is RLHF, not RL.
    spelled = _spell(abbreviation, expansion)
    return spelled is not None and spelled[1] >= len(expansion) - len(expansion.split()[-1])


def _is_plausible(pair: Pair) -> bool:
    if any(c.isdigit() for c in pair.expansion) or _NOT_AN_EXPANSION.match(pair.expansion):
        return False
    # "BERT (BERT-base)" names a variant of the model, not its long form. Whole words only:
    # "DA" is not a variant of "Data Augmentation".
    abbreviation, expansion = normal_form(pair.abbreviation), normal_form(pair.expansion)
    shorter, longer = sorted((abbreviation.split(), expansion.split()), key=len)
    return longer[: len(shorter)] != shorter


def _expansion_before(before: str, abbreviation: str) -> str | None:
    """The run of words before the parenthesis that the abbreviation spells best, then shortest.

    Words after the one holding the last letter are cut off: "Direct Preference Optimization
    algorithm (DPO)" is the name without "algorithm".
    """
    words = before.split()
    best, best_score = None, -1
    for count in range(1, min(_MAX_EXPANSION_WORDS, len(words)) + 1):
        run = words[-count:]
        # A comma or full stop ends the clause: words before it are not part of the name.
        if any(word.endswith(_CLAUSE_END) for word in run[:-1]):
            break
        candidate = " ".join(run).strip(_EDGE_PUNCTUATION)
        spelled = _spell(abbreviation, candidate) if candidate else None
        if spelled is not None and spelled[0] > best_score:
            best_score = spelled[0]
            tail = candidate.find(" ", spelled[1])
            best = (candidate if tail < 0 else candidate[:tail]).strip(_EDGE_PUNCTUATION)
    return best


def _spell(abbreviation: str, expansion: str) -> tuple[int, int] | None:
    """(letters on word starts, position of the last letter), or None if it does not spell it.

    The first letter starts the expansion, every capital starts a word (split at spaces and
    hyphens), lowercase letters may sit inside a word. All in the abbreviation's order. Among
    spellings the one with most letters on word starts wins, then the one ending latest.
    """
    letters = [c for c in abbreviation if c.isalpha()]
    text = expansion.lower()
    starts = [i == 0 or not expansion[i - 1].isalnum() for i in range(len(expansion))]
    if len(letters) < 2 or not text.startswith(letters[0].lower()):
        return None

    @cache
    def best(index: int, position: int) -> tuple[int, int] | None:
        if index == len(letters):
            return (0, position - 1)
        letter = letters[index]
        top = None
        for at in range(position, len(text)):
            if text[at] != letter.lower() or (letter.isupper() and not starts[at]):
                continue
            if index == 0 and at != 0:
                break
            rest = best(index + 1, at + 1)
            if rest is not None and (top is None or (rest[0] + starts[at], rest[1]) > top):
                top = (rest[0] + starts[at], rest[1])
        return top

    return best(0, 0)


def collect_candidates(connection: psycopg.Connection) -> Candidates:
    """Pairs and conflicts over every stored abstract.

    A conflict counts every pair of the corpus, covered by the dictionary or not. A pair is
    covered when both its normal forms are dictionary keys with one canonical name, and only
    uncovered pairs are candidates. A candidate takes the canonical name the dictionary already
    has for either of its keys.
    """
    known = dict(connection.execute("SELECT normal_form, canonical_name FROM synonym"))
    # abbreviation normal form -> expansion normal form -> spelling -> papers
    spellings: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    # Streamed: the corpus is every abstract collected, too many to hold at once.
    with connection.cursor() as cursor:
        for (abstract,) in cursor.stream("SELECT abstract FROM paper"):
            # A paper that spells one pair two ways still counts once.
            seen: set[tuple[str, str]] = set()
            for pair in find_pairs(abstract):
                key = (normal_form(pair.abbreviation), normal_form(pair.expansion))
                if key not in seen:
                    seen.add(key)
                    spellings[key[0]][key[1]][pair.expansion] += 1

    pairs: dict[str, tuple[str, int]] = {}
    conflicts: dict[str, dict[str, int]] = {}
    existing_names: dict[str, str] = {}
    for key, expansions in spellings.items():
        counted = {_most_common(spelled): sum(spelled.values()) for spelled in expansions.values()}
        entities, dictionary_entity = _entities(key, expansions, known)
        if len(entities | dictionary_entity) > 1:
            conflicts[key] = counted
            if dictionary_entity - entities:
                counted[known[key]] = 0  # the dictionary's reading, in no paper
        elif not all(
            key in known and expansion_key in known and known[key] == known[expansion_key]
            for expansion_key in expansions
        ):
            ((expansion, papers),) = counted.items()
            pairs[key] = (expansion, papers)
            (expansion_key,) = expansions
            if name := known.get(key) or known.get(expansion_key):
                existing_names[key] = name
    return Candidates(
        pairs=dict(sorted(pairs.items(), key=lambda item: (-item[1][1], item[0]))),
        conflicts=dict(sorted(conflicts.items())),
        existing_names=existing_names,
    )


def _entities(
    key: str, expansions: dict[str, Counter[str]], known: dict[str, str]
) -> tuple[set[str], set[str]]:
    """What an abbreviation stands for in the corpus and in the dictionary.

    Each is a set of normal forms of canonical names. A dictionary name equal to the
    abbreviation itself ("LoRA") says nothing about its expansion, unless an expansion is in
    the dictionary under another name.
    """
    corpus = {normal_form(known[e]) if e in known else e for e in expansions}
    if key not in known:
        return corpus, set()
    named = normal_form(known[key])
    if named != key or any(e in known for e in expansions):
        return corpus, {named}
    return corpus, set()


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
        canonical = candidates.existing_names.get(key, canonical)
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
