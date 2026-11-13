"""The normal form of a spelling and the synonym dictionary file that is loaded into the database.

The dictionary file is the source of truth (ADR 0002): a change to it recomputes canonical
names through the ``paper_entity`` view, with no extraction run.
"""

import random

import psycopg
import pytest

from research_radar import quality
from research_radar.normalize import normal_form
from research_radar.synonyms import SYNONYMS_FILE, load_synonyms, read_synonyms

DICTIONARY = """\
normal_form,canonical_name
rag,retrieval-augmented generation
retrieval augmented generation,retrieval-augmented generation
lora,LoRA
"""

NORMAL_FORM_EXAMPLES = [
    "Retrieval-Augmented  Generation",
    "retrieval augmented generation",
    "Retrieval–Augmented Generation",
    "GANs",
    "diffusion models",
    "bias",
    "Adam",
    "BERT",
    "Gaussian process",
    "Adagrad",
    "GPS",
    "GPT-4o",
    "Llama 3.1",
    "Low-Rank Adaptation.-.",
    "models.s",
    "glasses.s",
    "bosses.s",
    "GANs.",
    "LLMs,.",
    " -RAG- ",
]


def write_dictionary(tmp_path, text=DICTIONARY):
    path = tmp_path / "synonyms.csv"
    path.write_text(text, encoding="utf-8")
    return path


def table(connection) -> dict[str, str]:
    return dict(connection.execute("SELECT normal_form, canonical_name FROM synonym").fetchall())


# --- normal_form ---------------------------------------------------------------------------


def test_normal_form_lowercases_and_unifies_hyphens_and_spaces():
    spellings = [
        "Retrieval-Augmented  Generation",
        "retrieval augmented generation",
        "Retrieval–Augmented Generation",
    ]

    assert {normal_form(s) for s in spellings} == {"retrieval augmented generation"}


def test_normal_form_drops_plural_s_of_last_word():
    assert normal_form("GANs") == "gan"
    assert normal_form("diffusion models") == "diffusion model"


def test_plural_s_after_a_digit_is_dropped():
    assert normal_form("GPT4s") == "gpt4"
    assert normal_form("A100s") == "a100"
    assert normal_form("Llama2s") == "llama2"


def test_short_or_numeric_words_keep_their_s():
    assert normal_form("4s") == "4s"
    assert normal_form("A1s") == "a1s"
    assert normal_form("1990s") == "1990s"
    assert normal_form("100s") == "100s"


def test_leading_full_stop_is_kept():
    assert normal_form(".NET") == ".net"


def test_leading_colon_and_space_are_stripped():
    assert normal_form(": BERT") == "bert"


def test_normal_form_keeps_words_that_only_look_plural():
    assert normal_form("bias") == "bias"
    assert normal_form("Adam") == "adam"
    assert normal_form("BERT") == "bert"
    assert normal_form("Gaussian process") == "gaussian process"
    assert normal_form("Adagrad") == "adagrad"
    assert normal_form("GPS") == "gps"  # three letters: too short to judge


def test_normal_form_keeps_digits_and_versions():
    assert normal_form("GPT-4o") == "gpt 4o"
    assert normal_form("Llama 3.1") == "llama 3.1"


def test_normal_form_is_idempotent_on_examples():
    for spelling in NORMAL_FORM_EXAMPLES:
        assert normal_form(normal_form(spelling)) == normal_form(spelling), spelling


def test_normal_form_is_idempotent_on_random_text():
    rng = random.Random(20261113)
    alphabet = "abcAB s-.,;:–\t 0123456789  "
    for i in range(2000):
        # Every third spelling starts with a full stop: the leading dot is kept, so it must be
        # idempotent too.
        lead = "." if i % 3 == 0 else ""
        spelling = lead + "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14)))
        assert normal_form(normal_form(spelling)) == normal_form(spelling), spelling


def test_normal_form_drops_plural_s_once():
    assert normal_form("bosses") == "bosse"
    assert normal_form("bosse") == "bosse"


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("glasses.s", "glasses.s"),
        ("bosses.s", "bosses.s"),
        ("models.s", "models.s"),
        ("GANs.", "gan"),
        ("LLMs,.", "llm"),
        ("Low-Rank Adaptation.-.", "low rank adaptation"),
    ],
)
def test_normal_form_strips_edges_then_drops_plural_s_once(spelling, expected):
    assert normal_form(spelling) == expected
    assert normal_form(expected) == expected


def test_normal_form_of_nothing_is_empty():
    assert normal_form("") == ""
    assert normal_form(" - – — ") == ""
    assert normal_form(".,;:") == ""


# --- read_synonyms -------------------------------------------------------------------------


def test_read_synonyms_returns_normal_form_to_canonical_name(tmp_path):
    assert read_synonyms(write_dictionary(tmp_path)) == {
        "rag": "retrieval-augmented generation",
        "retrieval augmented generation": "retrieval-augmented generation",
        "lora": "LoRA",
    }


def test_read_synonyms_rejects_key_that_is_not_a_normal_form(tmp_path):
    path = write_dictionary(tmp_path, "normal_form,canonical_name\nRAG,retrieval-augmented\n")

    with pytest.raises(ValueError, match="RAG"):
        read_synonyms(path)


def test_read_synonyms_rejects_duplicate_key_and_empty_canonical_name(tmp_path):
    duplicate = write_dictionary(tmp_path, "normal_form,canonical_name\nrag,A\nrag,B\n")
    with pytest.raises(ValueError, match="rag"):
        read_synonyms(duplicate)

    empty = write_dictionary(tmp_path, "normal_form,canonical_name\nrag,\n")
    with pytest.raises(ValueError, match="rag"):
        read_synonyms(empty)


def test_header_only_file_is_an_empty_dictionary(tmp_path):
    assert read_synonyms(write_dictionary(tmp_path, "normal_form,canonical_name\n")) == {}


def test_shipped_synonyms_file_is_valid():
    assert isinstance(read_synonyms(SYNONYMS_FILE), dict)


# --- load_synonyms -------------------------------------------------------------------------


def test_load_synonyms_fills_table_from_file(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        count = load_synonyms(connection, write_dictionary(tmp_path))

        assert count == 3
        assert table(connection) == {
            "rag": "retrieval-augmented generation",
            "retrieval augmented generation": "retrieval-augmented generation",
            "lora": "LoRA",
        }


def test_load_synonyms_removes_entries_dropped_from_file(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write_dictionary(tmp_path))

        count = load_synonyms(
            connection, write_dictionary(tmp_path, "normal_form,canonical_name\nlora,LoRA\n")
        )
        assert (count, table(connection)) == (1, {"lora": "LoRA"})

        header_only = write_dictionary(tmp_path, "normal_form,canonical_name\n")
        assert load_synonyms(connection, header_only) == 0
        assert table(connection) == {}


def test_load_synonyms_is_all_or_nothing(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write_dictionary(tmp_path))
        before = table(connection)

        broken = write_dictionary(tmp_path, "normal_form,canonical_name\nlora,LoRA\nLoRA,x\n")
        with pytest.raises(ValueError):
            load_synonyms(connection, broken)

        assert table(connection) == before


def test_changing_dictionary_changes_canonical_names_without_extraction(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        run_id = connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash) VALUES ('test', 'test') RETURNING id"
        ).fetchone()[0]
        for arxiv_id, spelling in [("p0001", "RAG"), ("p0002", "retrieval-augmented generation")]:
            connection.execute(
                """
                INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                                   primary_category, first_version_date, arxiv_updated_at)
                VALUES (%s, 1, 't', 'a', 'h', '[]', 'cs.CL', '2026-10-01', '2026-10-01')
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
        assert [normal_form(s) for s in ("RAG", "retrieval-augmented generation")] == [
            "rag",
            "retrieval augmented generation",
        ]

        def names():
            return {row[0] for row in connection.execute("SELECT canonical_name FROM paper_entity")}

        def frozen():
            return (
                connection.execute("SELECT * FROM mention ORDER BY arxiv_id").fetchall(),
                connection.execute("SELECT * FROM paper_extraction ORDER BY arxiv_id").fetchall(),
            )

        before = frozen()
        assert names() == {"rag", "retrieval augmented generation"}

        load_synonyms(connection, write_dictionary(tmp_path))
        assert names() == {"retrieval-augmented generation"}

        changed = DICTIONARY.replace(
            "retrieval-augmented generation", "Retrieval-Augmented Generation"
        )
        load_synonyms(connection, write_dictionary(tmp_path, changed))
        assert names() == {"Retrieval-Augmented Generation"}

        assert frozen() == before


# --- frozen rules ---------------------------------------------------------------------------

# The version of the normal form rules: the same value quality.rules_sha256() writes to
# quality_measurement.rules_sha256. Stored normal forms were computed by these rules at
# extraction time and nothing recomputes them (ADR 0002, ticket #66).
NORMALIZE_SHA256 = "955592ee130427c74b1551707c0a2886e67fb6528d8bf776c2726c2f03887fec"


def test_normal_form_rules_are_frozen():
    assert quality.rules_sha256() == NORMALIZE_SHA256, (
        "правила нормальной формы заморожены (ADR 0002); менять только с решением разработчика "
        "и новым полным прогоном"
    )
