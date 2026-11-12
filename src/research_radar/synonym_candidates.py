"""Candidates for the synonym dictionary: "Long Form (ABBR)" pairs found in abstracts.

A candidate takes effect only when the developer moves the row into ``synonyms.csv`` (ADR 0002);
nothing here writes the dictionary or the ``synonym`` table. Grouping is by ``normal_form()``,
the key the dictionary itself uses, so "LoRA" and "LoRa" are one abbreviation. The candidates
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
from research_radar.synonyms import SYNONYMS_FILE, read_synonyms

_MAX_EXPANSION_WORDS = 8
_PARENTHESIS = re.compile(r"\(([^()]*)\)")
_BRACKETS = re.compile(r"\[[^\[\]]*\]")
_ABBREVIATION = re.compile(r"[A-Za-z-]{2,8}")
_QUOTES = "\"'“”‘’"
_CLAUSE_END = (".", ",", ";", ":", "!", "?")
# A bracket left over from an unbalanced or enclosing group, or a run of spaces where a group
# was blanked: the words around it are not one name.
_BROKEN = re.compile(r"[()\[\]{}]| {2}")
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
    # abbreviation to look at by hand -> {expansion or dictionary name: number of papers}
    conflicts: dict[str, dict[str, int]]


def find_pairs(abstract: str) -> list[Pair]:
    """Pairs written as "Long Form (ABBR)" or "ABBR (Long Form)", each once, in text order."""
    # arXiv abstracts carry line breaks and double spaces; after this a run of spaces can only
    # be a blanked group. Square-bracket citations are blanked like the parentheses below:
    # same length, so positions stay valid.
    text = _BRACKETS.sub(_blank, " ".join(abstract.split()))
    found: list[tuple[int, Pair]] = []
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
    inside = re.split(r"[;,]", match.group(1))[0].strip(" " + _QUOTES)
    before = blanked[: match.start()].rstrip()
    pair = None
    if _is_abbreviation(inside):
        if expansion := _expansion_before(before, inside):
            pair = Pair(expansion, inside)
    elif before and not _NOT_AN_EXPANSION.match(inside):
        abbreviation = before.split()[-1].strip(_QUOTES)
        words = len(inside.split())
        spelled = _spell(abbreviation, inside) if _is_abbreviation(abbreviation) else None
        # The last letter is in the last word: "RL (reinforcement learning from human
        # feedback)" is RLHF, not RL.
        if 0 < words <= _MAX_EXPANSION_WORDS and spelled and spelled[1] == words - 1:
            pair = Pair(inside, abbreviation)
    return pair if pair and _is_plausible(pair) else None


def _is_abbreviation(token: str) -> bool:
    return bool(_ABBREVIATION.fullmatch(token)) and sum(c.isupper() for c in token) >= 2


def _is_plausible(pair: Pair) -> bool:
    if any(c.isdigit() for c in pair.expansion) or _BROKEN.search(pair.expansion):
        return False
    # "BERT (BERT-base)" names a variant of the model, not its long form. Whole words only:
    # "DA" is not a variant of "Data Augmentation".
    abbreviation, expansion = normal_form(pair.abbreviation), normal_form(pair.expansion)
    shorter, longer = sorted((abbreviation.split(), expansion.split()), key=len)
    return longer[: len(shorter)] != shorter


def _expansion_before(before: str, abbreviation: str) -> str | None:
    """The run of words before the parenthesis that the abbreviation spells best, then shortest.

    Words after the one holding the last letter are cut off: "Direct Preference Optimization
    algorithm (DPO)" is the name without "algorithm". The run starts after the last bracket or
    blanked group, so it never spans one.
    """
    words = _BROKEN.split(before)[-1].split()
    best, best_hits = None, -1
    for count in range(1, min(_MAX_EXPANSION_WORDS, len(words)) + 1):
        run = words[-count:]
        # A comma or full stop ends the clause: words before it are not part of the name.
        if any(word.endswith(_CLAUSE_END) for word in run[:-1]):
            break
        candidate = " ".join(run).strip(_QUOTES)
        spelled = _spell(abbreviation, candidate)
        if spelled is not None and spelled[0] > best_hits:
            best_hits = spelled[0]
            best = " ".join(candidate.split()[: spelled[1] + 1]).strip(_QUOTES)
    return best


def _spell(abbreviation: str, expansion: str) -> tuple[int, int] | None:
    """(letters on word starts, index of the word holding the last letter), or None.

    The first letter starts the expansion, every capital starts a word (split at spaces and
    hyphens), lowercase letters may sit inside a word. All in the abbreviation's order. Among
    spellings the one with most letters on word starts wins, then the one ending latest.
    Positions are taken on the lowercased text itself: lowercasing "İ" adds a character.
    """
    letters = [c for c in abbreviation if c.isalpha()]
    text = expansion.lower()
    if len(letters) < 2 or not text.startswith(letters[0].lower()):
        return None
    starts = [i == 0 or text[i - 1] in " -" for i in range(len(text))]
    word_of, word = [], -1
    for i, c in enumerate(text):
        if c != " " and (i == 0 or text[i - 1] == " "):
            word += 1
        word_of.append(word)

    @cache
    def best(index: int, position: int) -> tuple[int, int] | None:
        if index == len(letters):
            return (0, word_of[position - 1])
        letter = letters[index]
        top = None
        for at in range(position, len(text) if index else 1):
            if text[at] != letter.lower() or (letter.isupper() and not starts[at]):
                continue
            rest = best(index + 1, at + 1)
            if rest is not None and (top is None or (rest[0] + starts[at], rest[1]) > top):
                top = (rest[0] + starts[at], rest[1])
        return top

    return best(0, 0)


def collect_candidates(
    connection: psycopg.Connection, synonyms_file: Path = SYNONYMS_FILE
) -> Candidates:
    """Pairs and conflicts over every stored abstract.

    The dictionary is the ``synonym`` table and ``synonyms_file`` together: rows moved into the
    file count before they are loaded, and a key the two name differently has both names.
    An abbreviation is covered, and left out, when it and all its readings in the corpus are
    dictionary keys under one canonical name. Otherwise it is a candidate when it has one
    reading, its key is no other abbreviation's reading, and neither key is a dictionary key or
    a key of an abbreviation shown by hand. Everything else is shown by hand in ``conflicts``,
    with the dictionary's names for its keys at 0 papers. So no candidate key is a dictionary
    key, and candidates share keys only through a common reading.
    """
    known: dict[str, set[str]] = defaultdict(set)
    table = connection.execute("SELECT normal_form, canonical_name FROM synonym")
    for key, name in [*table, *read_synonyms(synonyms_file).items()]:
        known[key].add(name)
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

    keys = {key: {key, *expansions} for key, expansions in spellings.items()}
    expansion_keys = {e for expansions in spellings.values() for e in expansions}
    covered = {
        key
        for key, group in keys.items()
        if group.issubset(known) and len(set().union(*(known[k] for k in group))) == 1
    }
    by_hand = {
        key
        for key, group in keys.items()
        if key not in covered
        and (
            len(spellings[key]) > 1
            or not group.isdisjoint(known)
            # "Lo-RA" is "lo ra", which may be the expansion of another abbreviation ("LR").
            or key in expansion_keys
            or any(k != normal_form(k) for k in group)
        )
    }
    # A key shared with an abbreviation shown by hand links the two: writing a name for one
    # and not the other would split the entity, so the link is followed to the end.
    shown = set().union(*(keys[key] for key in by_hand))
    while linked := {
        key
        for key, group in keys.items()
        if key not in covered | by_hand and not group.isdisjoint(shown)
    }:
        by_hand |= linked
        shown = shown.union(*(keys[key] for key in linked))

    pairs: dict[str, tuple[str, int]] = {}
    conflicts: dict[str, dict[str, int]] = {}
    for key, expansions in spellings.items():
        counted = {_most_common(spelled): sum(spelled.values()) for spelled in expansions.values()}
        if key in by_hand:
            named = set().union(*(known.get(k, set()) for k in keys[key])) - set(counted)
            conflicts[key] = counted | dict.fromkeys(sorted(named), 0)
        elif key not in covered:
            pairs[key] = next(iter(counted.items()))
    return Candidates(
        pairs=dict(sorted(pairs.items(), key=lambda item: (-item[1][1], item[0]))),
        conflicts=dict(sorted(conflicts.items())),
    )


def _most_common(spelled: Counter[str]) -> str:
    return min(spelled, key=lambda spelling: (-spelled[spelling], spelling))


def write_candidates_csv(candidates: Candidates, path: Path) -> None:
    """Pairs to ``path``; conflicts, one row per reading, to the ``.conflicts.csv`` beside it.

    A pair is two rows with one canonical name, the abbreviation's and the expansion's normal
    form: mentions of the expansion are matched by its own normal form, not the abbreviation's.
    A key already written keeps its row, and the pair's other key takes that row's name, so two
    abbreviations of one expansion share it. A pair whose two keys were already written under
    different names is left out.
    """
    pair_rows: list[tuple[str, str, int]] = []
    names: dict[str, str] = {}  # key -> canonical name already written for it
    for key, (expansion, papers) in candidates.pairs.items():
        pair_keys = dict.fromkeys((key, normal_form(expansion)))
        taken = {names[k] for k in pair_keys if k in names}
        if len(taken) > 1:
            continue
        name = taken.pop() if taken else expansion
        for k in pair_keys:
            if k not in names:
                names[k] = name
                pair_rows.append((k, name, papers))
    conflict_rows = [
        (key, name, papers)
        for key, readings in candidates.conflicts.items()
        for name, papers in sorted(readings.items(), key=lambda item: (-item[1], item[0]))
    ]
    _write(path, pair_rows)
    _write(path.with_suffix(".conflicts.csv"), conflict_rows)


def _write(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(_COLUMNS)
        writer.writerows(rows)
