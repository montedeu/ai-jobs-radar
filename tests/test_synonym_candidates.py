"""Candidates for the synonym dictionary: pairs "Long Form (ABBR)" found in abstracts."""

import csv
import re
from random import Random

import psycopg

from research_radar.normalize import normal_form
from research_radar.synonym_candidates import (
    Candidates,
    Pair,
    collect_candidates,
    find_pairs,
    write_candidates_csv,
)
from research_radar.synonyms import load_synonyms, read_synonyms

LORA_FULL = "We apply Low-Rank Adaptation (LoRA) to the model."
LORA_ABBREVIATION_FIRST = "LoRA (low rank adaptation) is cheap."
PEFT_EFFICIENT = "PEFT (parameter-efficient fine-tuning) saves memory."
PEFT_PROMPT = "PEFT (prompt engineering fine tuning) is another reading."
NLPROC = "We study natural language processing (NLProc)."
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


def test_keeps_pairs_whose_normal_form_is_only_a_substring_of_the_other():
    abstract = "Data Augmentation (DA), Relation Extraction (RE) and Task Arithmetic (TA)."

    assert find_pairs(abstract) == [
        Pair("Data Augmentation", "DA"),
        Pair("Relation Extraction", "RE"),
        Pair("Task Arithmetic", "TA"),
    ]


def test_ignores_expansion_with_words_after_the_last_letter():
    assert find_pairs("RL (reinforcement learning from human feedback) is used") == []


def test_cuts_words_after_the_last_letter_from_the_expansion_before_the_parenthesis():
    assert find_pairs("Direct Preference Optimization algorithm (DPO)") == [
        Pair("Direct Preference Optimization", "DPO")
    ]


def test_strips_quotes_around_the_expansion():
    assert find_pairs('a method called "Low-Rank Adaptation" (LoRA)') == [
        Pair("Low-Rank Adaptation", "LoRA")
    ]


def test_square_bracket_citations_are_not_part_of_the_pair():
    abstract = "supervised fine-tuning [12] (SFT) and Low-Rank Adaptation [3] (LoRA)"

    assert find_pairs(abstract) == [
        Pair("supervised fine-tuning", "SFT"),
        Pair("Low-Rank Adaptation", "LoRA"),
    ]


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


def test_expansion_broken_by_a_parenthesis_is_dropped():
    assert find_pairs("Retrieval (see A) augmented generation (RAG)") == []
    assert find_pairs("Retrieval (see (A)) augmented generation (RAG)") == []
    assert find_pairs("LoRA (low rank (x) adaptation) is cheap") == []


def test_unbalanced_parentheses_do_not_break_the_search():
    assert find_pairs("Low-Rank Adaptation (LoRA") == []
    assert find_pairs("Retrieval ( augmented generation (RAG)") == []
    assert find_pairs("((LoRA)") == []
    assert find_pairs("x) Low-Rank Adaptation (LoRA)) and (") == [
        Pair("Low-Rank Adaptation", "LoRA")
    ]


def test_line_breaks_inside_an_expansion_do_not_drop_it():
    assert find_pairs("Low-Rank\n  Adaptation  (LoRA)") == [Pair("Low-Rank Adaptation", "LoRA")]


def test_lowercasing_that_adds_a_character_does_not_shift_positions():
    assert find_pairs("İstanbul Speech Corpus (ISC)") == [Pair("İstanbul Speech Corpus", "ISC")]


def test_empty_text_has_no_pairs():
    assert find_pairs("") == []
    assert find_pairs("()") == []


def test_pointer_words_are_filtered_only_inside_the_parenthesis():
    assert find_pairs("Including Expansions (IE) helps") == [Pair("Including Expansions", "IE")]


def test_any_text_gives_pairs_without_brackets_or_runs_of_spaces():
    pieces = ["Low", "Rank", "Adaptation", "LoRA", "İstanbul", "ISC", "see", "RAG", "Retrieval"]
    pieces += ["(", ")", "[", "]", "[1]", ",", ";", ".", '"', "-", "  ", "\n", "e.g.", "x2"]
    pieces += ["İstanbul Speech Corpus", "Low-Rank Adaptation", "augmented generation"]
    pieces += ["(LoRA)", "(RAG", "(ISC)", "(see (A))", "(low rank (x) adaptation)", "(e.g. x)"]
    random = Random(39)
    found = 0
    for _ in range(3000):
        text = " ".join(random.choice(pieces) for _ in range(random.randint(0, 12)))
        found += len(pairs := find_pairs(text))
        for pair in pairs:
            assert not re.search(r"[()\[\]{}]| {2}|\d", pair.expansion), text
    assert found > 20


# --- collect_candidates --------------------------------------------------------------------


def test_one_pair_per_paper_counts_once(database_url):
    abstract = "Low-Rank Adaptation (LoRA) works. We use Low-Rank Adaptation (LoRA) again."

    candidates = collect(database_url, [abstract])

    assert candidates.pairs == {"lora": ("Low-Rank Adaptation", 1)}


def test_one_paper_with_two_spellings_of_a_pair_counts_once(database_url):
    abstract = "Low-Rank Adaptation (LoRA) works. low-rank adaptation (LoRA) again."

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


def test_pair_with_only_the_abbreviation_in_synonym_table_goes_to_conflicts(database_url):
    candidates = collect(database_url, [LORA_FULL], synonyms=[("lora", "LoRA")])

    assert candidates.pairs == {}
    assert candidates.conflicts == {"lora": {"Low-Rank Adaptation": 1, "LoRA": 0}}


def test_pair_with_only_the_expansion_in_synonym_table_goes_to_conflicts(database_url):
    synonyms = [("low rank adaptation", "LRA")]

    candidates = collect(database_url, [LORA_FULL], synonyms=synonyms)

    assert candidates.pairs == {}
    assert candidates.conflicts == {"lora": {"Low-Rank Adaptation": 1, "LRA": 0}}


def test_pair_whose_two_sides_have_different_names_is_a_conflict(database_url):
    covered = [("lora", "LoRA"), ("low rank adaptation", "Low-Rank Adaptation")]

    candidates = collect(database_url, [LORA_FULL], synonyms=covered)

    assert candidates.pairs == {}
    assert candidates.conflicts == {"lora": {"Low-Rank Adaptation": 1, "LoRA": 0}}


def test_names_that_differ_only_in_case_do_not_cover_a_pair(database_url):
    # paper_entity groups by the name as written, so these are two entities.
    covered = [("lora", "LoRA"), ("low rank adaptation", "Lora")]

    candidates = collect(database_url, [LORA_FULL], synonyms=covered)

    assert candidates.pairs == {}
    assert candidates.conflicts == {"lora": {"Low-Rank Adaptation": 1, "LoRA": 0, "Lora": 0}}


def test_conflict_counts_pairs_the_dictionary_covers(database_url):
    covered = [("peft", "PEFT"), ("parameter efficient fine tuning", "PEFT")]

    candidates = collect(database_url, [PEFT_EFFICIENT, PEFT_PROMPT], synonyms=covered)

    assert candidates.pairs == {}
    assert candidates.conflicts == {
        "peft": {
            "parameter-efficient fine-tuning": 1,
            "prompt engineering fine tuning": 1,
            "PEFT": 0,
        }
    }


def test_two_expansions_the_dictionary_names_alike_are_a_conflict(database_url):
    synonyms = [
        ("parameter efficient fine tuning", "PEFT"),
        ("prompt engineering fine tuning", "PEFT"),
    ]

    candidates = collect(database_url, [PEFT_EFFICIENT, PEFT_PROMPT], synonyms=synonyms)

    assert candidates.pairs == {}
    assert candidates.conflicts == {
        "peft": {
            "parameter-efficient fine-tuning": 1,
            "prompt engineering fine tuning": 1,
            "PEFT": 0,
        }
    }


def test_two_expansions_the_dictionary_fully_covers_are_skipped(database_url):
    synonyms = [
        ("peft", "PEFT"),
        ("parameter efficient fine tuning", "PEFT"),
        ("prompt engineering fine tuning", "PEFT"),
    ]

    candidates = collect(database_url, [PEFT_EFFICIENT, PEFT_PROMPT], synonyms=synonyms)

    assert candidates == Candidates(pairs={}, conflicts={})


def test_abbreviation_sharing_an_expansion_with_a_conflict_is_not_a_candidate(database_url):
    abstracts = ["NLP (Natural Language Processing) works.", NLPROC]

    candidates = collect(database_url, abstracts, synonyms=[("nlp", "NLP")])

    assert candidates.pairs == {}
    assert candidates.conflicts == {
        "nlp": {"Natural Language Processing": 1, "NLP": 0},
        "nlproc": {"natural language processing": 1},
    }


def test_pair_whose_key_is_not_a_normal_form_goes_to_conflicts(database_url):
    # normal_form("Low-Rank Adaptation.-.") is "low rank adaptation.", which read_synonyms
    # rejects as a key because its own normal form drops the full stop.
    candidates = collect(database_url, ["Low-Rank Adaptation.-. (LoRA)"])

    assert candidates.pairs == {}
    assert candidates.conflicts == {"lora": {"Low-Rank Adaptation.-.": 1}}


def test_dictionary_expansion_against_another_one_in_the_corpus_is_a_conflict(database_url):
    synonyms = [("rag", "retrieval-augmented generation")]

    candidates = collect(
        database_url, ["RAG (red-team augmented generation) is a test."], synonyms=synonyms
    )

    assert candidates.pairs == {}
    assert candidates.conflicts == {
        "rag": {"red-team augmented generation": 1, "retrieval-augmented generation": 0}
    }


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


def test_pair_whose_keys_were_written_under_two_names_is_left_out(tmp_path):
    candidates = Candidates(
        pairs={"aa": ("Foo Bar", 2), "bb": ("Baz Qux", 2), "foo bar": ("Baz Qux", 1)},
        conflicts={},
    )
    path = tmp_path / "candidates.csv"

    write_candidates_csv(candidates, path)

    assert read_rows(path) == [
        HEADER,
        ["aa", "Foo Bar", "2"],
        ["foo bar", "Foo Bar", "2"],
        ["bb", "Baz Qux", "2"],
        ["baz qux", "Baz Qux", "2"],
    ]


# --- moving candidate rows into the dictionary ---------------------------------------------


def transfer(database_url, tmp_path, dictionary, abstracts):
    """Collect against ``dictionary``, append every candidate row to it, load the result.

    Returns the merged dictionary as ``read_synonyms`` reads it, which raises on a repeated key.
    """
    candidates = collect(database_url, abstracts, synonyms=dictionary)
    path = tmp_path / "candidates.csv"
    write_candidates_csv(candidates, path)
    merged_file = tmp_path / "synonyms.csv"
    with merged_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["normal_form", "canonical_name"])
        writer.writerows(dictionary)
        writer.writerows(row[:2] for row in read_rows(path)[1:])
    merged = read_synonyms(merged_file)
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, merged_file)
    assert {key: merged[key] for key, _ in dictionary} == dict(dictionary)
    return candidates, merged


def entity_names(database_url, spellings):
    """spelling -> canonical name that ``paper_entity`` gives a mention of it."""
    with psycopg.connect(database_url) as connection:
        run_id = connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash) VALUES ('test', 'test') RETURNING id"
        ).fetchone()[0]
        for number, spelling in enumerate(spellings):
            arxiv_id = f"m{number:04d}"
            connection.execute(
                """
                INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                                   primary_category, first_version_date, arxiv_updated_at)
                VALUES (%s, 1, 't', '', 'h', '[]', 'cs.CL', '2026-10-01', '2026-10-01')
                """,
                (arxiv_id,),
            )
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
        rows = connection.execute("SELECT arxiv_id, canonical_name FROM paper_entity").fetchall()
    by_paper = dict(rows)
    return {spelling: by_paper[f"m{n:04d}"] for n, spelling in enumerate(spellings)}


def test_candidate_rows_moved_over_the_dictionary_keep_it_whole(database_url, tmp_path):
    dictionary = [
        ("nlp", "NLP"),
        ("rag", "RAG"),
        ("direct preference optimization", "DPO"),
        ("parameter efficient fine tuning", "PEFT"),
        ("prompt engineering fine tuning", "PEFT"),
    ]
    abstracts = [
        LORA_FULL,
        LORA_ABBREVIATION_FIRST,
        "Supervised Fine-Tuning (SFT) first.",
        "supervised fine-tuning (SuFT) again.",
        "NLP (Natural Language Processing) works.",
        NLPROC,
        "Retrieval-Augmented Generation (RAG) helps.",
        "Direct Preference Optimization (DPO) aligns.",
        PEFT_EFFICIENT,
        PEFT_PROMPT,
    ]

    candidates, merged = transfer(database_url, tmp_path, dictionary, abstracts)
    names = entity_names(
        database_url,
        ["LoRA", "low rank adaptation", "SFT", "SuFT", "Supervised Fine-Tuning", "NLP"],
    )

    assert set(candidates.pairs) == {"lora", "sft", "suft"}
    assert names["LoRA"] == names["low rank adaptation"] == "Low-Rank Adaptation"
    assert names["SFT"] == names["SuFT"] == names["Supervised Fine-Tuning"]
    assert names["NLP"] == "NLP"
    # NLProc shares its expansion with NLP, which the dictionary has: no row names either.
    assert "natural language processing" not in merged
    assert "nlproc" not in merged


def test_candidate_rows_from_any_corpus_load_over_the_dictionary(database_url, tmp_path):
    phrases = [
        "Low-Rank Adaptation (LoRA)",
        "LoRA (low rank adaptation)",
        "Low Rank (LR)",
        "Rank Adaptation (RA)",
        "Retrieval Augmented Generation (RAG)",
        "Retrieval-Augmented Generation (RAGen)",
        "Natural Language Processing (NLP)",
        "natural language processing (NLProc)",
        "Language Processing (LP)",
        "LP (low precision)",
        "Low-Precision (LP)",
        "İstanbul Speech Corpus (ISC)",
        "Speech Corpus (SC)",
        "(see Section 3)",
        "((",
        "))",
        "[1]",
        ".",
    ]
    dictionary = [("lora", "LoRA"), ("language processing", "LP"), ("speech corpus", "SC")]
    random = Random(42)
    abstracts = [
        " ".join(random.choice(phrases) for _ in range(random.randint(0, 6))) for _ in range(200)
    ]

    candidates, merged = transfer(database_url, tmp_path, dictionary, abstracts)

    assert candidates.pairs
    for key, (expansion, _) in candidates.pairs.items():
        assert merged[key] == merged[normal_form(expansion)]
