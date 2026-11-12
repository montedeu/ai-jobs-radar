"""Candidates for the synonym dictionary: pairs "Long Form (ABBR)" found in abstracts."""

import csv

import psycopg

from research_radar.normalize import normal_form
from research_radar.synonym_candidates import (
    Candidates,
    Pair,
    collect_candidates,
    find_pairs,
    write_candidates_csv,
)
from research_radar.synonyms import load_synonyms

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
        Pair("Low-Rank Adaptation", "LoRA"),
        Pair("retrieval-augmented generation", "RAG"),
    ]


def test_allows_lowercase_letters_inside_the_abbreviation():
    abstract = "multilingual Bidirectional Encoder Representations from Transformers (mBERT)"

    assert find_pairs(abstract) == [
        Pair("multilingual Bidirectional Encoder Representations from Transformers", "mBERT")
    ]
    assert find_pairs("a single capital (Lora) is not enough") == []


def test_drops_expansion_longer_than_eight_words():
    assert find_pairs("One two three four five six seven eight Nine (ON)") == []


def test_expansion_stops_at_the_previous_clause():
    assert find_pairs("We study memory. Parameter-efficient fine-tuning (PEFT) works.") == [
        Pair("Parameter-efficient fine-tuning", "PEFT")
    ]


def test_expansion_is_not_cut_short_of_a_capital_letter():
    assert find_pairs("Proximal Policy Optimization (PPO) is stable.") == [
        Pair("Proximal Policy Optimization", "PPO")
    ]


def test_capital_letters_must_start_words_of_the_expansion():
    assert find_pairs("Largelanguage model (LLM)") == []


def test_ignores_expansion_that_is_a_reference_or_has_digits():
    abstract = (
        "SFT (see Fig. 2 and Table 3), LLMs (like all models), "
        "DPO (direct preference optimization 2), RL (such as reinforcement learning)."
    )

    assert find_pairs(abstract) == []


def test_ignores_model_variants():
    abstract = "GPT-4 (GPT-4o), LLaMA (LLaMA-2) and BERT (BERT-base)."

    assert find_pairs(abstract) == []


def test_finds_pair_inside_enclosing_parenthesis():
    assert find_pairs("tuning (low-rank adaptation (LoRA)) works") == [
        Pair("low-rank adaptation", "LoRA")
    ]


def test_reads_abbreviation_up_to_a_citation_separator():
    assert find_pairs("low-rank adaptation (LoRA; Hu et al., 2021)") == [
        Pair("low-rank adaptation", "LoRA")
    ]
    assert find_pairs("low-rank adaptation (LoRA, Hu et al.)") == [
        Pair("low-rank adaptation", "LoRA")
    ]


def test_parenthesis_inside_the_expansion_is_skipped():
    assert find_pairs("Retrieval (see A) augmented generation (RAG)") == [
        Pair("Retrieval augmented generation", "RAG")
    ]


# --- collect_candidates --------------------------------------------------------------------


def test_one_pair_per_paper_counts_once(database_url):
    abstract = "Low-Rank Adaptation (LoRA) works. We use Low-Rank Adaptation (LoRA) again."

    candidates = collect(database_url, [abstract])

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 1)}


def test_pairs_are_grouped_by_normal_form(database_url):
    candidates = collect(database_url, [LORA_FULL, "Low-Rank Adaptation (LoRa) is common."])

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
    covered = [("lora", "LoRA"), ("low rank adaptation", "LoRA")]

    candidates = collect(database_url, [LORA_FULL, LORA_ABBREVIATION_FIRST], synonyms=covered)

    assert candidates == Candidates(pairs={}, conflicts={})


def test_pair_with_only_the_abbreviation_in_synonym_table_is_still_a_candidate(database_url):
    candidates = collect(database_url, [LORA_FULL], synonyms=[("lora", "LoRA")])

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 1)}


def test_pair_whose_two_sides_have_different_names_is_still_a_candidate(database_url):
    covered = [("lora", "LoRA"), ("low rank adaptation", "Low-Rank Adaptation")]

    candidates = collect(database_url, [LORA_FULL], synonyms=covered)

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 1)}


def test_conflict_is_shown_when_only_the_abbreviation_is_in_synonym_table(database_url):
    candidates = collect(database_url, [PEFT_EFFICIENT, PEFT_PROMPT], synonyms=[("peft", "PEFT")])

    assert set(candidates.conflicts) == {"peft"}


def test_reads_abstracts_through_a_streaming_cursor():
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, query):
            assert "paper" in query
            yield (LORA_FULL,)

    class Connection:
        def execute(self, query):
            assert "FROM synonym" in query  # reading every abstract at once goes through here
            return []

        def cursor(self):
            return Cursor()

    assert collect_candidates(Connection()).pairs == {"lora": ("Low-Rank Adaptation", 1)}


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

    assert read_rows(path) == [
        HEADER,
        ["lora", "Low-Rank Adaptation", "2"],
        ["low rank adaptation", "Low-Rank Adaptation", "2"],
    ]
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


def test_candidate_rows_make_both_spellings_one_entity(database_url, tmp_path):
    candidates = collect(database_url, [LORA_FULL, LORA_ABBREVIATION_FIRST])
    path = tmp_path / "candidates.csv"
    write_candidates_csv(candidates, path)
    dictionary = tmp_path / "synonyms.csv"
    with dictionary.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows(row[:2] for row in read_rows(path))

    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, dictionary)
        run_id = connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash) VALUES ('test', 'test') RETURNING id"
        ).fetchone()[0]
        for arxiv_id, spelling in [("p0000", "LoRA"), ("p0001", "Low-Rank Adaptation")]:
            connection.execute(
                """
                INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,
                                              output_tokens)
                VALUES (%s, %s, 'h', 1, 1)
                """,
                (run_id, arxiv_id),
            )
            connection.execute(
                "INSERT INTO mention VALUES (%s, %s, 'method', %s, %s)",
                (run_id, arxiv_id, spelling, normal_form(spelling)),
            )
        names = {row[0] for row in connection.execute("SELECT canonical_name FROM paper_entity")}

    assert names == {"Low-Rank Adaptation"}


def test_two_abbreviations_of_one_expansion_write_its_row_once(tmp_path):
    candidates = Candidates(
        pairs={
            "nlp": ("Natural Language Processing", 3),
            "nlproc": ("natural language processing", 1),
        },
        conflicts={},
    )
    path = tmp_path / "candidates.csv"

    write_candidates_csv(candidates, path)

    assert read_rows(path) == [
        HEADER,
        ["nlp", "Natural Language Processing", "3"],
        ["natural language processing", "Natural Language Processing", "3"],
        ["nlproc", "Natural Language Processing", "1"],
    ]
