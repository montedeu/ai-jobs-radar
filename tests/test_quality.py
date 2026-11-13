import hashlib
import importlib.util
import json
from datetime import date

import psycopg
import pytest

from research_radar import cli, quality
from research_radar.extraction import start_run
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS, EXTRACTION_MODEL

GOLD = [
    {
        "arxiv_id": "2511.00001",
        "mentions": [
            {"entity_type": "method", "spelling": "LoRA"},
            {"entity_type": "model", "spelling": "Llama 3.1 8B"},
            {"entity_type": "task", "spelling": "machine translation"},
        ],
    },
    {"arxiv_id": "2511.00002", "mentions": []},
    {"arxiv_id": "2511.00003", "mentions": [{"entity_type": "dataset", "spelling": "FLORES-200"}]},
]
EXTRACTED = {
    "2511.00001": [("method", "LoRA"), ("model", "Llama-3.1 8B"), ("task", "translation")],
    "2511.00002": [("method", "survey")],
    "2511.00003": [("model", "FLORES-200")],
}


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(connection, arxiv_id):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%s, 1, 't', 'a', 'h', '[]', 'cs.CL', %s, '2025-12-01')
        """,
        (arxiv_id, date(2025, 11, 3)),
    )


def extract(connection, run_id, arxiv_id, mentions, *, stored_form=None):
    """Completion mark and mentions as raw SQL; stored_form overrides the saved normal form."""
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, %s, 'h', 1, 1)",
        (run_id, arxiv_id),
    )
    for entity_type, spelling in mentions:
        connection.execute(
            "INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form)"
            " VALUES (%s, %s, %s, %s, %s)",
            (run_id, arxiv_id, entity_type, spelling, stored_form or spelling.lower()),
        )


def gold_papers(rows=GOLD):
    return [quality.GoldPaper.model_validate(row) for row in rows]


def one_paper(*mentions):
    return gold_papers(
        [
            {
                "arxiv_id": "2511.00001",
                "mentions": [{"entity_type": t, "spelling": s} for t, s in mentions],
            }
        ]
    )


@pytest.fixture
def world(connection):
    for row in GOLD:
        add_paper(connection, row["arxiv_id"])
    run = start_run(connection, model=EXTRACTION_MODEL)
    for arxiv_id, mentions in EXTRACTED.items():
        extract(connection, run, arxiv_id, mentions)
    return run


@pytest.fixture
def single(connection):
    """One paper, one run; the test adds the mentions."""
    add_paper(connection, "2511.00001")
    return start_run(connection, model=EXTRACTION_MODEL)


def write_gold(path, rows=GOLD):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def measure(connection, run, gold=None, *, sha="g", label="before"):
    gold = gold_papers() if gold is None else gold
    return quality.measure(connection, run, gold, gold_sha256=sha, label=label)


def logged(connection):
    return connection.execute("SELECT count(*) FROM quality_measurement").fetchone()[0]


def test_read_gold_rejects_unknown_type_with_line_number(tmp_path):
    bad = {"arxiv_id": "2511.00002", "mentions": [{"entity_type": "metric", "spelling": "F1"}]}
    path = write_gold(tmp_path / "gold.jsonl", [GOLD[0], bad])
    with pytest.raises(ValueError, match="line 2"):
        quality.read_gold(path)
    blank = {"arxiv_id": "2511.00002", "mentions": [{"entity_type": "task", "spelling": " "}]}
    with pytest.raises(ValueError, match="line 1"):
        quality.read_gold(write_gold(tmp_path / "blank.jsonl", [blank]))
    (tmp_path / "broken.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        quality.read_gold(tmp_path / "broken.jsonl")


def test_read_gold_rejects_duplicate_paper(tmp_path):
    path = write_gold(tmp_path / "gold.jsonl", [GOLD[0], GOLD[1], GOLD[0]])
    with pytest.raises(ValueError, match=r"line 3.*2511\.00001"):
        quality.read_gold(path)


def test_read_gold_reads_papers_without_mentions(tmp_path):
    papers = quality.read_gold(write_gold(tmp_path / "gold.jsonl"))
    assert [p.arxiv_id for p in papers] == ["2511.00001", "2511.00002", "2511.00003"]
    assert papers[1].mentions == []


def test_perfect_extraction_scores_one(connection, single):
    extract(connection, single, "2511.00001", [("method", "LoRA"), ("task", "translation")])
    gold = one_paper(("method", "lora"), ("task", "Translation"))
    result = measure(connection, single, gold)
    for score in (result.overall_typed, result.overall_untyped):
        assert (score.precision, score.recall) == (1.0, 1.0)
        assert (score.true_positives, score.predicted, score.gold) == (2, 2, 2)
    assert result.by_type["method"].precision == 1.0
    assert result.by_type["dataset"].precision is None


def test_precision_and_recall_per_type_with_counts(connection, world):
    result = measure(connection, world)
    assert set(result.by_type) == {"method", "model", "task", "dataset"}
    method = result.by_type["method"]
    model = result.by_type["model"]
    task = result.by_type["task"]
    dataset = result.by_type["dataset"]
    assert (method.precision, method.recall) == (0.5, 1.0)
    assert (method.true_positives, method.predicted, method.gold) == (1, 2, 1)
    assert (model.precision, model.recall) == (0.5, 1.0)
    assert (task.precision, task.recall) == (0.0, 0.0)
    assert (task.true_positives, task.predicted, task.gold) == (0, 1, 1)
    assert dataset.precision is None
    assert dataset.recall == 0.0
    assert (dataset.true_positives, dataset.predicted, dataset.gold) == (0, 0, 1)


def test_type_confusion_counts_in_untyped_but_not_in_typed(connection, world):
    result = measure(connection, world)
    untyped, typed = result.overall_untyped, result.overall_typed
    assert (untyped.true_positives, untyped.predicted, untyped.gold) == (3, 5, 4)
    assert (typed.true_positives, typed.predicted, typed.gold) == (2, 5, 4)
    assert untyped.precision == 0.6
    assert untyped.recall == 0.75


def test_dictionary_applies_to_both_sides(connection, single):
    extract(connection, single, "2511.00001", [("method", "RAG")])
    connection.execute(
        "INSERT INTO synonym VALUES ('rag', 'RAG'), ('retrieval augmented generation', 'RAG')"
    )
    connection.execute("INSERT INTO synonym_version (file_sha256, entries) VALUES ('dict', 2)")
    gold = one_paper(("method", "retrieval-augmented generation"))
    result = measure(connection, single, gold)
    assert result.overall_typed.true_positives == 1
    assert result.overall_typed.precision == 1.0
    assert result.dictionary_sha256 == "dict"


def test_both_sides_use_todays_normal_form(connection, single):
    # Stored under the rules of an earlier version: the plural s was kept.
    extract(connection, single, "2511.00001", [("method", "Adapters")], stored_form="adapters")
    result = measure(connection, single, one_paper(("method", "adapter")))
    assert (result.overall_typed.true_positives, result.overall_typed.predicted) == (1, 1)


def test_gold_spelling_with_empty_normal_form_is_dropped(connection, single):
    extract(connection, single, "2511.00001", [("method", "LoRA")])
    result = measure(connection, single, one_paper(("method", "LoRA"), ("method", "...")))
    assert result.overall_typed.gold == 1


def test_two_spellings_of_one_entity_count_once(connection, single):
    extract(connection, single, "2511.00001", [("method", "LoRA"), ("method", "lora")])
    result = measure(connection, single, one_paper(("method", "LoRA")))
    assert (result.overall_typed.predicted, result.overall_typed.true_positives) == (1, 1)


def test_gold_paper_without_completion_mark_stops_the_measurement(connection, world):
    add_paper(connection, "2511.00004")
    gold = gold_papers([*GOLD, {"arxiv_id": "2511.00004", "mentions": []}])
    with pytest.raises(ValueError, match="2511.00004"):
        measure(connection, world, gold)
    assert logged(connection) == 0


def test_gold_paper_missing_from_the_database_stops_the_measurement(connection, world):
    gold = gold_papers([*GOLD, {"arxiv_id": "2511.00077", "mentions": []}])
    with pytest.raises(ValueError, match="2511.00077"):
        measure(connection, world, gold)
    assert logged(connection) == 0


def test_empty_gold_stops_the_measurement(connection, world):
    with pytest.raises(ValueError, match="empty"):
        measure(connection, world, [])
    assert logged(connection) == 0


def test_unknown_label_and_run_are_refused(connection, world):
    with pytest.raises(ValueError, match="label"):
        measure(connection, world, label="during")
    with pytest.raises(ValueError, match="run"):
        measure(connection, world + 100)
    assert logged(connection) == 0


def test_measure_refuses_a_connection_with_a_transaction_open(database_url, world):
    with psycopg.connect(database_url) as open_transaction:
        open_transaction.execute("SELECT 1")
        with pytest.raises(ValueError, match="transaction"):
            measure(open_transaction, world)


def test_measurement_is_logged_with_run_dictionary_and_gold_hash(connection, world):
    connection.execute("INSERT INTO synonym_version (file_sha256, entries) VALUES ('dict', 0)")
    result = measure(connection, world, sha="gold-hash", label="after")
    run_id, dictionary, gold_hash, label, metrics, measured_at = connection.execute(
        "SELECT run_id, dictionary_sha256, gold_sha256, label, metrics, measured_at"
        " FROM quality_measurement"
    ).fetchone()
    assert (run_id, dictionary, gold_hash, label) == (world, "dict", "gold-hash", "after")
    assert measured_at is not None
    assert result.run_id == world
    assert metrics["by_type"]["task"] == {
        "precision": 0.0,
        "recall": 0.0,
        "true_positives": 0,
        "predicted": 1,
        "gold": 1,
    }
    assert metrics["by_type"]["dataset"]["precision"] is None
    assert metrics["overall_untyped"]["true_positives"] == 3


def test_measurement_without_a_dictionary_version_logs_null(connection, world):
    result = measure(connection, world)
    assert result.dictionary_sha256 is None
    row = connection.execute("SELECT dictionary_sha256 FROM quality_measurement").fetchone()
    assert row == (None,)


def test_report_shows_latest_before_and_after_with_counts(connection, world):
    assert "no measurements" in quality.report(connection)
    measure(connection, world, sha="oldgold-0123456789", label="before")
    other = gold_papers(
        [{"arxiv_id": "2511.00002", "mentions": [{"entity_type": "method", "spelling": "survey"}]}]
    )
    measure(connection, world, other, sha="midgold-0123456789", label="after")
    measure(connection, world, other, sha="newgold-0123456789", label="after")
    text = quality.report(connection)
    assert "before" in text and "after" in text
    assert "oldgold-0123" in text
    assert "newgold-0123" in text
    assert "midgold-0123" not in text  # only the latest measurement of a label
    assert "method" in text and "dataset" in text
    assert "precision 0.50 (1/2)" in text
    assert "recall 0.75 (3/4)" in text
    assert "precision n/a (0/0)" in text
    assert "cs.CL" in text and "one annotator" in text


def test_cli_measure_and_measurements(
    world, connection, database_url, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("DATABASE_URL", database_url)
    path = write_gold(tmp_path / "gold.jsonl")
    cli.main(["measure", "--run", str(world), "--gold", str(path), "--label", "before"])
    row = connection.execute("SELECT gold_sha256, label FROM quality_measurement").fetchone()
    assert row == (hashlib.sha256(path.read_bytes()).hexdigest(), "before")
    capsys.readouterr()
    cli.main(["measurements"])
    assert "before" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["measure", "--run", str(world), "--gold", str(path), "--label", "middle"])


def test_cli_measure_reports_a_bad_gold_file(world, database_url, monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", database_url)
    path = tmp_path / "gold.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 1"):
        cli.main(["measure", "--run", str(world), "--gold", str(path), "--label", "before"])


def test_gold_paper_with_a_stale_abstract_stops_the_measurement(connection, world):
    connection.execute("UPDATE paper SET abstract_hash = 'changed' WHERE arxiv_id = '2511.00002'")
    with pytest.raises(ValueError, match="2511.00002"):
        measure(connection, world)
    assert logged(connection) == 0


def test_dictionary_key_by_old_rules_stops_the_measurement(connection, world):
    connection.execute("INSERT INTO synonym VALUES ('Adapters', 'adapter'), ('lora', 'LoRA')")
    with pytest.raises(ValueError, match="Adapters.*load-synonyms"):
        measure(connection, world)
    assert logged(connection) == 0


def test_measurement_logs_the_hash_of_the_normal_form_rules(
    connection, world, monkeypatch, tmp_path
):
    measure(connection, world)
    source = tmp_path / "normalize_other.py"
    source.write_text("def normal_form(spelling):\n    return spelling\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("normalize_other", source)
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    monkeypatch.setattr(quality, "normalize", other)
    measure(connection, world)
    first, second = (
        row[0]
        for row in connection.execute("SELECT rules_sha256 FROM quality_measurement ORDER BY id")
    )
    assert first and second and first != second


def test_measure_leaves_the_isolation_level_as_it_was(connection, world):
    before = connection.isolation_level
    measure(connection, world)
    assert connection.isolation_level == before


def test_report_pairs_before_with_the_after_of_the_same_basis(connection, world):
    measure(connection, world, sha="oldgold-0123456789", label="before")
    connection.execute("INSERT INTO synonym_version (file_sha256, entries) VALUES ('dict2', 0)")
    measure(connection, world, sha="newgold-0123456789", label="after")
    text = quality.report(connection)
    assert "different basis" in text
    assert "oldgold-0123" in text and "newgold-0123" in text
    measure(connection, world, sha="samegold-012345678", label="before")
    text = quality.report(connection)
    assert "different basis" not in text
    assert "samegold-012" in text
    assert "oldgold-0123" not in text


def test_report_survives_old_log_rows(connection, world):
    connection.execute(
        "INSERT INTO quality_measurement (run_id, gold_sha256, label, rules_sha256, metrics)"
        " VALUES (%s, 'oldoldoldoldold', 'before', 'r',"
        ' \'{"by_type": {"metric": {"precision": 1.0}}, "overall_typed": {}}\')',
        (world,),
    )
    text = quality.report(connection)
    assert "metric" in text and "oldoldoldold" in text


def test_measure_prints_the_report_format(world, database_url, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DATABASE_URL", database_url)
    path = write_gold(tmp_path / "gold.jsonl")
    cli.main(["measure", "--run", str(world), "--gold", str(path), "--label", "before"])
    out = capsys.readouterr().out
    assert "precision 0.50 (1/2)" in out
    assert "precision n/a (0/0)" in out
    assert "recall 0.75 (3/4)" in out
    assert "one annotator" in out


def test_parse_gold_reads_the_bytes_it_is_given():
    raw = b'{"arxiv_id": "2511.00001", "mentions": []}\n{not json\n'
    with pytest.raises(ValueError, match="line 2"):
        quality.parse_gold(raw, "gold.jsonl")


def give_up(connection, run, arxiv_id, *, abstract_hash="h", count=EXTRACTION_MAX_ATTEMPTS):
    """Collected batches that ended with an invalid answer for the paper, count times."""
    for _ in range(count):
        batch_id = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'collected', 0) RETURNING id",
            (run,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash, outcome)"
            " VALUES (%s, %s, %s, 'invalid')",
            (batch_id, arxiv_id, abstract_hash),
        )


def gold_with_given_up_paper(connection):
    """GOLD plus 2511.00004 with two mentions, a paper the test may have the run give up on."""
    add_paper(connection, "2511.00004")
    return gold_papers(
        [
            *GOLD,
            {
                "arxiv_id": "2511.00004",
                "mentions": [
                    {"entity_type": "method", "spelling": "adapters"},
                    {"entity_type": "task", "spelling": "parsing"},
                ],
            },
        ]
    )


def test_gold_paper_the_run_gave_up_on_counts_as_an_empty_prediction(connection, world):
    gold = gold_with_given_up_paper(connection)
    give_up(connection, world, "2511.00004")
    result = measure(connection, world, gold)
    # The same 2 of 5 predicted hit as without the paper; its 2 mentions are only missed.
    assert (result.overall_typed.predicted, result.overall_typed.true_positives) == (5, 2)
    assert result.overall_typed.precision == 2 / 5
    assert (result.overall_typed.gold, result.overall_typed.recall) == (6, 2 / 6)
    assert result.given_up_papers == ("2511.00004",)


def test_given_up_papers_are_logged_and_reported(connection, world):
    gold = gold_with_given_up_paper(connection)
    give_up(connection, world, "2511.00004")
    result = measure(connection, world, gold)
    assert "given up: 1 (2511.00004)" in quality.format_measurement(result)
    metrics = connection.execute("SELECT metrics FROM quality_measurement").fetchone()[0]
    assert metrics["given_up_papers"] == ["2511.00004"]
    assert "given up: 1 (2511.00004)" in quality.report(connection)


def test_measurement_without_given_up_papers_says_zero(connection, world):
    result = measure(connection, world)
    assert result.given_up_papers == ()
    assert "given up: 0" in quality.format_measurement(result)


def test_paper_without_a_mark_and_not_given_up_still_stops_the_measurement(connection, world):
    gold = gold_with_given_up_paper(connection)
    give_up(connection, world, "2511.00004", count=EXTRACTION_MAX_ATTEMPTS - 1)
    with pytest.raises(ValueError, match="2511.00004"):
        measure(connection, world, gold)
    assert logged(connection) == 0


def test_attempts_on_an_old_abstract_are_not_a_give_up(connection, world):
    gold = gold_with_given_up_paper(connection)
    give_up(connection, world, "2511.00004", abstract_hash="old")
    with pytest.raises(ValueError, match="2511.00004"):
        measure(connection, world, gold)
    assert logged(connection) == 0


def test_report_reads_old_rows_without_given_up_papers(connection, world):
    connection.execute(
        "INSERT INTO quality_measurement (run_id, gold_sha256, label, rules_sha256, metrics)"
        " VALUES (%s, 'oldoldoldoldold', 'before', 'r', '{}')",
        (world,),
    )
    assert "given up: 0" in quality.report(connection)


def test_given_up_paper_with_mentions_of_an_old_abstract_predicts_nothing(connection, world):
    gold = gold_with_given_up_paper(connection)
    # Extracted on the old abstract, then the abstract changed and the run gave up on the new.
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, '2511.00004', 'old', 1, 1)",
        (world,),
    )
    for kind, spelling in [("method", "adapters"), ("task", "parsing")]:
        connection.execute(
            "INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form)"
            " VALUES (%s, '2511.00004', %s, %s, %s)",
            (world, kind, spelling, spelling),
        )
    give_up(connection, world, "2511.00004")
    result = measure(connection, world, gold)
    assert (result.overall_typed.predicted, result.overall_typed.true_positives) == (5, 2)
    assert result.given_up_papers == ("2511.00004",)


def test_paper_with_attempts_and_a_current_mark_is_not_given_up(connection, world):
    gold = gold_with_given_up_paper(connection)
    extract(connection, world, "2511.00004", [("method", "adapters")])
    give_up(connection, world, "2511.00004")
    result = measure(connection, world, gold)
    assert result.given_up_papers == ()
    assert (result.overall_typed.predicted, result.overall_typed.true_positives) == (6, 3)
