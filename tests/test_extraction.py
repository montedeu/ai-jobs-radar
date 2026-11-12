from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest
from pydantic import ValidationError

from research_radar import extraction
from research_radar.extraction import (
    BudgetExceeded,
    ExtractionRequest,
    ExtractionResult,
    collect,
    spent_usd,
    start_run,
    submit,
)
from research_radar.mentions import Mention, PaperMentions
from research_radar.papers import abstract_hash
from research_radar.settings import (
    BATCH_INPUT_USD_PER_MTOK,
    BATCH_OUTPUT_USD_PER_MTOK,
    EXTRACTION_MAX_OUTPUT_TOKENS,
    EXTRACTION_MODEL,
)

LORA = ExtractionRequest(
    "2511.00001",
    "Adapting a translator",
    "We fine-tune Llama 3.1 8B with Low-Rank Adaptation (LoRA) for machine translation, "
    "and evaluate on FLORES-200.",
)
SURVEY = ExtractionRequest("2511.00002", "A survey", "We survey open problems in the field.")

LORA_ANSWER = ExtractionResult(
    "2511.00001",
    PaperMentions(
        mentions=[
            Mention(entity_type="model", spelling="Llama 3.1 8B"),
            Mention(entity_type="method", spelling="LoRA", expansion="Low-Rank Adaptation"),
            Mention(entity_type="task", spelling="machine translation"),
            Mention(entity_type="dataset", spelling="FLORES-200"),
        ]
    ),
    input_tokens=120,
    output_tokens=40,
)
SURVEY_ANSWER = ExtractionResult("2511.00002", PaperMentions(mentions=[]), 90, 5)


def usd(input_tokens, output_tokens):
    return (
        Decimal(input_tokens) * BATCH_INPUT_USD_PER_MTOK
        + Decimal(output_tokens) * BATCH_OUTPUT_USD_PER_MTOK
    ) / 1_000_000


class FakeModel:
    """Stands in for the model client: serves the answers it was given, and logs calls."""

    def __init__(self, *answers, ended=True, counted_tokens=1000):
        self.answers = {a.arxiv_id: a for a in answers}
        self.ended = ended
        self.counted_tokens = counted_tokens
        self.calls = []
        self.batches = {}

    def count_tokens(self, requests):
        self.calls.append("count_tokens")
        return self.counted_tokens

    def submit_batch(self, requests):
        self.calls.append("submit_batch")
        batch_id = f"batch-{len(self.batches) + 1}"
        self.batches[batch_id] = [r.arxiv_id for r in requests]
        return batch_id

    def batch_ended(self, batch_id):
        self.calls.append("batch_ended")
        return self.ended

    def batch_results(self, batch_id):
        self.calls.append("batch_results")
        return iter(self.answers[i] for i in self.batches[batch_id] if i in self.answers)


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url) as connection:
        yield connection


def add_papers(connection, *requests):
    for request in requests:
        connection.execute(
            """
            INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                               primary_category, first_version_date, arxiv_updated_at)
            VALUES (%s, 1, %s, %s, %s, '[]', 'cs.CL', '2025-11-03', '2025-11-03')
            """,
            (request.arxiv_id, request.title, request.abstract, abstract_hash(request.abstract)),
        )
    connection.commit()


def new_run(connection):
    return start_run(connection, model="fake-model", prompt_hash="stub")


def run_batch(connection, model, run_id, *requests):
    batch_id = submit(connection, model, run_id, requests)
    return batch_id, collect(connection, model, batch_id)


def mention_rows(connection):
    return connection.execute(
        "SELECT arxiv_id, entity_type, spelling, normal_form, expansion FROM mention"
        " ORDER BY arxiv_id, entity_type, spelling"
    ).fetchall()


def test_mention_schema_rejects_unknown_type():
    with pytest.raises(ValidationError):
        Mention(entity_type="person", spelling="Alice")
    with pytest.raises(ValidationError):
        PaperMentions.model_validate({"mentions": [{"entity_type": "x", "spelling": "y"}]})


def test_mention_schema_keeps_spelling_and_expansion():
    mention = Mention(entity_type="method", spelling=" LoRA ", expansion="Low-Rank Adaptation")
    assert (mention.spelling, mention.expansion) == (" LoRA ", "Low-Rank Adaptation")
    assert Mention(entity_type="task", spelling="NER").expansion is None


def test_mention_schema_forbids_extra_fields():
    # Structured outputs need additionalProperties: false on every object.
    schema = PaperMentions.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Mention"]["additionalProperties"] is False


def test_model_is_called_only_through_the_client(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER)
    batch_id = submit(connection, model, new_run(connection), [LORA])
    assert model.calls == ["count_tokens", "submit_batch"]
    collect(connection, model, batch_id)
    assert model.calls[2:] == ["batch_ended", "batch_results"]


def test_submit_stores_batch_state_in_database(connection):
    add_papers(connection, LORA, SURVEY)
    run_id = new_run(connection)
    batch_id = submit(connection, FakeModel(), run_id, [LORA, SURVEY])
    row = connection.execute(
        "SELECT run_id, provider_batch_id, status, estimated_cost_usd, collected_at"
        " FROM extraction_batch WHERE id = %s",
        (batch_id,),
    ).fetchone()
    assert row[:3] == (run_id, "batch-1", "submitted")
    assert row[3] > 0
    assert row[4] is None
    papers = connection.execute(
        "SELECT arxiv_id, abstract_hash FROM extraction_batch_paper WHERE batch_id = %s"
        " ORDER BY arxiv_id",
        (batch_id,),
    ).fetchall()
    assert papers == [
        (LORA.arxiv_id, abstract_hash(LORA.abstract)),
        (SURVEY.arxiv_id, abstract_hash(SURVEY.abstract)),
    ]


def test_estimate_prices_input_and_the_largest_possible_output(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel(counted_tokens=2_000_000)
    batch_id = submit(connection, model, new_run(connection), [LORA, SURVEY])
    stored = connection.execute(
        "SELECT estimated_cost_usd FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone()[0]
    assert stored == usd(2_000_000, 2 * EXTRACTION_MAX_OUTPUT_TOKENS)


def test_collect_waits_for_unfinished_batch(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel(LORA_ANSWER, SURVEY_ANSWER, ended=False)
    _, counter = run_batch(connection, model, new_run(connection), LORA, SURVEY)
    assert counter == Counter(pending=2)
    assert "batch_results" not in model.calls
    assert connection.execute("SELECT count(*) FROM paper_extraction").fetchone() == (0,)
    assert connection.execute("SELECT status FROM extraction_batch").fetchone() == ("submitted",)


def test_collect_stores_mentions_with_normal_form_and_expansion(connection):
    add_papers(connection, LORA)
    _, counter = run_batch(connection, FakeModel(LORA_ANSWER), new_run(connection), LORA)
    assert counter == Counter(extracted=1, mentions=4)
    assert mention_rows(connection) == [
        ("2511.00001", "dataset", "FLORES-200", "flores 200", None),
        ("2511.00001", "method", "LoRA", "lora", "Low-Rank Adaptation"),
        ("2511.00001", "model", "Llama 3.1 8B", "llama 3.1 8b", None),
        ("2511.00001", "task", "machine translation", "machine translation", None),
    ]
    assert connection.execute("SELECT status FROM extraction_batch").fetchone() == ("collected",)


def test_paper_without_mentions_gets_completion_mark(connection):
    add_papers(connection, SURVEY)
    run_id = new_run(connection)
    run_batch(connection, FakeModel(SURVEY_ANSWER), run_id, SURVEY)
    assert connection.execute(
        "SELECT run_id, arxiv_id, abstract_hash FROM paper_extraction"
    ).fetchall() == [(run_id, SURVEY.arxiv_id, abstract_hash(SURVEY.abstract))]
    assert mention_rows(connection) == []


def test_token_usage_is_stored_per_paper_and_summed_per_run(connection):
    add_papers(connection, LORA, SURVEY)
    run_id = new_run(connection)
    run_batch(connection, FakeModel(LORA_ANSWER, SURVEY_ANSWER), run_id, LORA, SURVEY)
    per_paper = connection.execute(
        "SELECT arxiv_id, input_tokens, output_tokens FROM paper_extraction ORDER BY arxiv_id"
    ).fetchall()
    assert per_paper == [("2511.00001", 120, 40), ("2511.00002", 90, 5)]
    assert connection.execute(
        "SELECT sum(input_tokens), sum(output_tokens) FROM paper_extraction WHERE run_id = %s",
        (run_id,),
    ).fetchone() == (210, 45)
    assert spent_usd(connection) == usd(210, 45)


def test_unreceived_batch_counts_as_spent_by_its_estimate(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER, ended=False)
    batch_id = submit(connection, model, new_run(connection), [LORA])
    estimate = connection.execute("SELECT estimated_cost_usd FROM extraction_batch").fetchone()[0]
    assert spent_usd(connection) == estimate
    model.ended = True
    collect(connection, model, batch_id)
    assert spent_usd(connection) == usd(120, 40)


def test_batch_over_ceiling_is_not_submitted(connection):
    add_papers(connection, LORA)
    model = FakeModel(counted_tokens=30_000_000)  # $1.50 at the batch input price
    with pytest.raises(BudgetExceeded):
        submit(connection, model, new_run(connection), [LORA], ceiling_usd=Decimal(1))
    assert "submit_batch" not in model.calls
    assert connection.execute("SELECT count(*) FROM extraction_batch").fetchone() == (0,)


def test_batch_is_counted_against_already_spent_money(connection):
    add_papers(connection, LORA, SURVEY)
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, %s, 'h', 19_999_000, 0)",  # $0.99995 of the $1 ceiling
        (new_run(connection), SURVEY.arxiv_id),
    )
    connection.commit()
    ceiling = Decimal(1)
    model = FakeModel(counted_tokens=1000)
    run_id = new_run(connection)
    # On its own the batch is far below the ceiling.
    assert usd(1000, EXTRACTION_MAX_OUTPUT_TOKENS) < Decimal("0.001")
    with pytest.raises(BudgetExceeded):
        submit(connection, model, run_id, [LORA], ceiling_usd=ceiling)
    assert "submit_batch" not in model.calls
    assert submit(connection, model, run_id, [LORA], ceiling_usd=Decimal(2)) is not None


def test_reextracting_a_paper_replaces_its_mentions_in_the_same_run(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER)
    run_id = new_run(connection)
    run_batch(connection, model, run_id, LORA)
    edited = ExtractionRequest(LORA.arxiv_id, LORA.title, "We study LoRA for summarization.")
    connection.execute(
        "UPDATE paper SET abstract = %s, abstract_hash = %s",
        (edited.abstract, abstract_hash(edited.abstract)),
    )
    connection.commit()
    model.answers[LORA.arxiv_id] = ExtractionResult(
        LORA.arxiv_id,
        PaperMentions(mentions=[Mention(entity_type="task", spelling="summarization")]),
        50,
        10,
    )
    _, counter = run_batch(connection, model, run_id, edited)
    assert counter == Counter(extracted=1, mentions=1)
    assert mention_rows(connection) == [
        ("2511.00001", "task", "summarization", "summarization", None)
    ]
    # Both batches were paid for, so the paper's usage is their sum.
    assert connection.execute(
        "SELECT abstract_hash, input_tokens, output_tokens FROM paper_extraction"
    ).fetchall() == [(abstract_hash(edited.abstract), 170, 50)]


def test_spellings_of_one_entity_are_separate_rows_and_one_paper_entity(connection):
    add_papers(connection, LORA)
    answer = ExtractionResult(
        LORA.arxiv_id,
        PaperMentions(
            mentions=[
                Mention(entity_type="method", spelling="LoRA"),
                Mention(entity_type="method", spelling="lora"),
                Mention(entity_type="method", spelling="LoRA", expansion="Low-Rank Adaptation"),
            ]
        ),
        10,
        5,
    )
    run_batch(connection, FakeModel(answer), new_run(connection), LORA)
    assert [row[2:] for row in mention_rows(connection)] == [
        ("LoRA", "lora", "Low-Rank Adaptation"),
        ("lora", "lora", None),
    ]
    assert connection.execute("SELECT count(*) FROM paper_entity").fetchone() == (1,)


def test_answer_not_matching_the_schema_gets_no_mark_and_others_are_kept(connection):
    add_papers(connection, LORA, SURVEY)
    broken = ExtractionResult(LORA.arxiv_id, None, 120, 40)
    model = FakeModel(broken, SURVEY_ANSWER)
    _, counter = run_batch(connection, model, new_run(connection), LORA, SURVEY)
    assert counter == Counter(extracted=1, failed=1)
    assert connection.execute("SELECT arxiv_id FROM paper_extraction").fetchall() == [
        (SURVEY.arxiv_id,)
    ]
    # The unusable answer was still billed.
    assert spent_usd(connection) == usd(210, 45)


def test_paper_missing_from_results_is_counted_failed(connection):
    add_papers(connection, LORA, SURVEY)
    _, counter = run_batch(connection, FakeModel(SURVEY_ANSWER), new_run(connection), LORA, SURVEY)
    assert counter == Counter(extracted=1, failed=1)


def test_collecting_a_collected_batch_changes_nothing(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER)
    batch_id, _ = run_batch(connection, model, new_run(connection), LORA)
    calls_before = len(model.calls)
    assert collect(connection, model, batch_id) == Counter()
    assert len(model.calls) == calls_before
    assert len(mention_rows(connection)) == 4


def test_empty_set_does_not_touch_the_client(connection):
    model = FakeModel()
    assert submit(connection, model, new_run(connection), []) is None
    assert model.calls == []


def test_paper_already_in_the_run_with_the_same_abstract_is_skipped(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel(LORA_ANSWER, SURVEY_ANSWER)
    run_id = new_run(connection)
    run_batch(connection, model, run_id, LORA)
    submit(connection, model, run_id, [LORA, SURVEY])
    assert model.batches["batch-2"] == [SURVEY.arxiv_id]


def test_paper_in_an_unreceived_batch_is_not_sent_again(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER, ended=False)
    run_id = new_run(connection)
    submit(connection, model, run_id, [LORA])
    assert submit(connection, model, run_id, [LORA]) is None
    assert list(model.batches) == ["batch-1"]


def test_test_without_fake_client_fails_instead_of_calling_api():
    with pytest.raises(AssertionError, match="Anthropic API"):
        extraction.AnthropicModelClient()
    anthropic = pytest.importorskip("anthropic")
    with pytest.raises(AssertionError, match="Anthropic API"):
        anthropic.Anthropic(api_key="not-a-key")


def batch_entry(result_type, text=None, usage=(100, 20)):
    message = SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]),
    )
    result = SimpleNamespace(type=result_type, message=message)
    return SimpleNamespace(custom_id="2511.00001", result=result)


def test_batch_request_asks_for_the_mention_schema_on_the_extraction_model():
    params = extraction.request_params(LORA)
    assert params["model"] == EXTRACTION_MODEL
    assert LORA.abstract in params["messages"][0]["content"]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["output_config"]["format"]["schema"]["additionalProperties"] is False


def test_batch_entry_is_read_into_a_result_with_its_tokens():
    entry = batch_entry("succeeded", '{"mentions": [{"entity_type": "task", "spelling": "NER"}]}')
    assert extraction.result_from_batch_entry(entry) == ExtractionResult(
        "2511.00001", PaperMentions(mentions=[Mention(entity_type="task", spelling="NER")]), 100, 20
    )


def test_batch_entry_not_in_the_schema_keeps_its_tokens_but_no_mentions():
    for text in ('{"mentions": [{"entity_type": "x", "spelling": "y"}]}', "not json", None):
        result = extraction.result_from_batch_entry(batch_entry("succeeded", text))
        assert (result.mentions, result.input_tokens, result.output_tokens) == (None, 100, 20)


@pytest.mark.parametrize("result_type", ["errored", "canceled", "expired"])
def test_batch_entry_that_was_not_answered_is_skipped(result_type):
    assert extraction.result_from_batch_entry(batch_entry(result_type)) is None
