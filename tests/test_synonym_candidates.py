"""Candidates for the synonym dictionary: pairs "Long Form (ABBR)" found in abstracts."""

import csv

import psycopg

from research_radar.synonym_candidates import (
    Candidates,
    Pair,
    collect_candidates,
    find_pairs,
    write_candidates_csv,
)

LORA_FULL = "We apply Low-Rank Adaptation (LoRA) to the model."
LORA_ABBREVIATION_FIRST = "LoRA (low rank adaptation) is cheap."
PEFT_EFFICIENT = "PEFT (parameter-efficient fine-tuning) saves memory."
PEFT_PROMPT = "PEFT (prompt engineering fine tuning) is another reading."
HEADER = ["normal_form", "canonical_name", "papers"]


def insert_papers(connection, abstracts):
    for number, abstract in enumerate(abstracts):
        connection.execute(
            """
            INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                               primary_category, first_version_date, arxiv_updated_at)
            VALUES (%s, 1, 't', %s, 'h', '[]', 'cs.CL', '2026-10-01', '2026-10-01')
            """,
            (f"p{number:04d}", abstract),
        )


def collect(database_url, abstracts, synonyms=()):
    with psycopg.connect(database_url) as connection:
        insert_papers(connection, abstracts)
        for key, name in synonyms:
            connection.execute("INSERT INTO synonym VALUES (%s, %s)", (key, name))
        return collect_candidates(connection)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.reader(file))


# --- find_pairs ----------------------------------------------------------------------------


def test_finds_expansion_followed_by_abbreviation():
    assert find_pairs("We apply Low-Rank Adaptation (LoRA) to the model.") == [
        Pair("Low-Rank Adaptation", "LoRA")
    ]


def test_finds_abbreviation_followed_by_expansion():
    assert find_pairs("RAG (retrieval-augmented generation) helps.") == [
        Pair("retrieval-augmented generation", "RAG")
    ]


def test_ignores_parenthesis_that_is_not_an_abbreviation():
    assert find_pairs("The setup (see Section 3) uses n samples (n = 5) (e.g. BERT).") == []


def test_ignores_pair_whose_letters_do_not_match():
    assert find_pairs("we train the model (GPT)") == []


def test_finds_pairs_with_nested_parentheses_and_several_pairs():
    abstract = "Low-Rank Adaptation (LoRA (see Section 2)) and retrieval-augmented generation (RAG)"

    assert find_pairs(abstract) == [
        Pair("retrieval-augmented generation", "RAG"),
        Pair("Low-Rank Adaptation", "LoRA"),
    ]


def test_allows_lowercase_letters_inside_the_abbreviation():
    assert find_pairs("multilingual BERT (mBERT) is used") == [Pair("multilingual BERT", "mBERT")]
    assert find_pairs("a single capital (Lora) is not enough") == []


def test_drops_expansion_longer_than_eight_words():
    long_form = "one two three four five six seven eight nine"

    assert find_pairs(f"{long_form} (OTTFFSSEN)") == []


def test_expansion_stops_at_the_previous_clause():
    assert find_pairs("We study memory. Parameter-efficient fine-tuning (PEFT) works.") == [
        Pair("Parameter-efficient fine-tuning", "PEFT")
    ]


# --- collect_candidates --------------------------------------------------------------------


def test_one_pair_per_paper_counts_once(database_url):
    abstract = "Low-Rank Adaptation (LoRA) works. We use Low-Rank Adaptation (LoRA) again."

    candidates = collect(database_url, [abstract])

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 1)}


def test_pairs_are_grouped_by_normal_form(database_url):
    candidates = collect(database_url, [LORA_FULL, "Low-Rank Adaptation (LORA) is common."])

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 2)}
    assert candidates.conflicts == {}


def test_abbreviation_with_two_expansions_is_a_conflict(database_url):
    abstracts = [LORA_FULL, LORA_ABBREVIATION_FIRST, PEFT_EFFICIENT, PEFT_PROMPT]

    candidates = collect(database_url, abstracts)

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 2)}
    assert candidates.conflicts == {
        "peft": {"parameter-efficient fine-tuning": 1, "prompt engineering fine tuning": 1}
    }


def test_same_expansion_spelled_differently_is_not_a_conflict(database_url):
    abstracts = [
        "Retrieval-Augmented Generation (RAG) is used.",
        "We use retrieval augmented generation (RAG).",
    ]

    candidates = collect(database_url, abstracts)

    assert candidates.conflicts == {}
    assert candidates.pairs == {"rag": ("Retrieval-Augmented Generation", 2)}


def test_pair_already_in_synonym_table_is_skipped(database_url):
    abstracts = [LORA_FULL, PEFT_EFFICIENT, PEFT_PROMPT]

    candidates = collect(database_url, abstracts, synonyms=[("lora", "LoRA"), ("peft", "PEFT")])

    assert candidates == Candidates(pairs={}, conflicts={})


def test_empty_corpus_gives_empty_candidates(database_url):
    assert collect(database_url, []) == Candidates(pairs={}, conflicts={})


# --- write_candidates_csv ------------------------------------------------------------------


def test_write_candidates_csv_and_conflicts_file(tmp_path):
    candidates = Candidates(
        pairs={"lora": ("Low-Rank Adaptation", 2)},
        conflicts={"peft": {"parameter-efficient fine-tuning": 3, "prompt, engineering": 1}},
    )
    path = tmp_path / "candidates.csv"

    write_candidates_csv(candidates, path)

    assert read_rows(path) == [HEADER, ["lora", "Low-Rank Adaptation", "2"]]
    assert read_rows(tmp_path / "candidates.conflicts.csv") == [
        HEADER,
        ["peft", "parameter-efficient fine-tuning", "3"],
        ["peft", "prompt, engineering", "1"],
    ]


def test_write_candidates_csv_for_empty_corpus_writes_headers_only(tmp_path):
    path = tmp_path / "candidates.csv"

    write_candidates_csv(Candidates(pairs={}, conflicts={}), path)

    assert read_rows(path) == [HEADER]
    assert read_rows(tmp_path / "candidates.conflicts.csv") == [HEADER]
