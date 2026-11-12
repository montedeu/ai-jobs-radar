import re
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

import psycopg
import pytest
from pydantic import ValidationError

from research_radar import extraction
from research_radar.extraction import (
    BatchRejected,
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
    return extraction._usd(EXTRACTION_MODEL, input_tokens, output_tokens)


class FakeModel:
    """Stands in for the model client: serves the answers it was given, and logs calls."""

    def __init__(
        self, *answers, ended=True, counted_tokens=1000, fail_submit=False, reject_submit=False
    ):
        self.answers = {a.arxiv_id: a for a in answers}
        self.batch_answers = {}  # batch id -> {arxiv_id: answer}, when batches differ
        self.fail_submit = fail_submit
        self.reject_submit = reject_submit
        self.models = []  # the model each call was given
        self.sent = []  # the requests of each submitted batch
        self.ended = ended
        self.counted_tokens = counted_tokens
        self.calls = []
        self.batches = {}

    def count_tokens(self, requests, *, model):
        self.calls.append("count_tokens")
        self.models.append(model)
        return self.counted_tokens

    def submit_batch(self, requests, *, model):
        self.calls.append("submit_batch")
        self.models.append(model)
        if self.fail_submit:
            raise RuntimeError("connection lost")
        if self.reject_submit:
            raise BatchRejected("invalid request")
        self.sent.append(list(requests))
        batch_id = f"batch-{len(self.batches) + 1}"
        self.batches[batch_id] = [r.arxiv_id for r in requests]
        return batch_id

    def batch_ended(self, batch_id):
        self.calls.append("batch_ended")
        return self.ended

    def batch_results(self, batch_id):
        self.calls.append("batch_results")
        answers = self.batch_answers.get(batch_id, self.answers)
        return iter(answers[i] for i in self.batches[batch_id] if i in answers)


@pytest.fixture
def connection(database_url):
    # Autocommit: submit and collect need a connection with no transaction open, and a
    # test's own read must not leave one behind.
    with psycopg.connect(database_url, autocommit=True) as connection:
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
    return start_run(connection, model=EXTRACTION_MODEL, prompt_hash="stub")


def run_batch(connection, model, run_id, *requests):
    batch_id = submit(connection, model, run_id, [r.arxiv_id for r in requests])
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
    batch_id = submit(connection, model, new_run(connection), [LORA.arxiv_id])
    assert model.calls == ["count_tokens", "submit_batch"]
    collect(connection, model, batch_id)
    assert model.calls[2:] == ["batch_ended", "batch_results"]


def test_submit_stores_batch_state_in_database(connection):
    add_papers(connection, LORA, SURVEY)
    run_id = new_run(connection)
    batch_id = submit(connection, FakeModel(), run_id, [LORA.arxiv_id, SURVEY.arxiv_id])
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
    batch_id = submit(connection, model, new_run(connection), [LORA.arxiv_id, SURVEY.arxiv_id])
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
    batch_id = submit(connection, model, new_run(connection), [LORA.arxiv_id])
    estimate = connection.execute("SELECT estimated_cost_usd FROM extraction_batch").fetchone()[0]
    assert spent_usd(connection) == estimate
    model.ended = True
    collect(connection, model, batch_id)
    assert spent_usd(connection) == usd(120, 40)


def test_batch_over_ceiling_is_not_submitted(connection):
    add_papers(connection, LORA)
    model = FakeModel(counted_tokens=30_000_000)  # $1.50 at the batch input price
    with pytest.raises(BudgetExceeded):
        submit(connection, model, new_run(connection), [LORA.arxiv_id], ceiling_usd=Decimal(1))
    assert "submit_batch" not in model.calls
    assert connection.execute("SELECT count(*) FROM extraction_batch").fetchone() == (0,)


def test_batch_is_counted_against_already_spent_money(connection):
    add_papers(connection, LORA)
    connection.execute(
        "INSERT INTO extraction_batch (run_id, provider_batch_id, status, estimated_cost_usd,"
        " actual_cost_usd) VALUES (%s, 'earlier', 'collected', 2, %s)",
        (new_run(connection), Decimal("0.99995")),  # of the $1 ceiling
    )
    model = FakeModel(counted_tokens=1000)
    run_id = new_run(connection)
    # On its own the batch is far below the ceiling.
    assert usd(1000, EXTRACTION_MAX_OUTPUT_TOKENS) < Decimal("0.001")
    with pytest.raises(BudgetExceeded):
        submit(connection, model, run_id, [LORA.arxiv_id], ceiling_usd=Decimal(1))
    assert "submit_batch" not in model.calls
    assert submit(connection, model, run_id, [LORA.arxiv_id], ceiling_usd=Decimal(2)) is not None


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
    submit(connection, model, run_id, [LORA.arxiv_id, SURVEY.arxiv_id])
    assert model.batches["batch-2"] == [SURVEY.arxiv_id]


def test_paper_in_an_unreceived_batch_is_not_sent_again(connection):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER, ended=False)
    run_id = new_run(connection)
    submit(connection, model, run_id, [LORA.arxiv_id])
    assert submit(connection, model, run_id, [LORA.arxiv_id]) is None
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


def test_batch_request_asks_for_the_mention_schema_on_the_given_model():
    params = extraction.request_params(LORA, "some-model")
    assert params["model"] == "some-model"
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


def advisory_locks(connection):
    query = "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'"
    return connection.execute(query).fetchone()[0]


def test_batch_row_is_written_before_the_batch_is_sent(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel(fail_submit=True)
    with pytest.raises(RuntimeError):
        submit(connection, model, new_run(connection), [LORA.arxiv_id, SURVEY.arxiv_id])
    status, provider_id, estimate = connection.execute(
        "SELECT status, provider_batch_id, estimated_cost_usd FROM extraction_batch"
    ).fetchone()
    assert (status, provider_id) == ("sending", None)
    assert connection.execute("SELECT count(*) FROM extraction_batch_paper").fetchone() == (2,)
    assert spent_usd(connection) == estimate
    # Nobody can tell whether the batch left, so collect leaves it for a person.
    batch_id = connection.execute("SELECT id FROM extraction_batch").fetchone()[0]
    healthy = FakeModel(LORA_ANSWER)
    assert collect(connection, healthy, batch_id) == Counter(unsent=2)
    assert healthy.calls == []


def test_sending_batch_blocks_the_same_papers(connection):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    with pytest.raises(RuntimeError):
        submit(connection, FakeModel(fail_submit=True), run_id, [LORA.arxiv_id])
    healthy = FakeModel()
    assert submit(connection, healthy, run_id, [LORA.arxiv_id]) is None
    assert healthy.calls == []


def test_paper_missing_from_the_database_fails_before_the_client_is_called(connection):
    model = FakeModel()
    with pytest.raises(ValueError, match="2511.00001"):
        submit(connection, model, new_run(connection), [LORA.arxiv_id])
    assert model.calls == []
    assert connection.execute("SELECT count(*) FROM extraction_batch").fetchone() == (0,)


def test_unknown_model_fails_before_the_client_is_called(connection):
    add_papers(connection, LORA)
    run_id = start_run(connection, model="no-such-model", prompt_hash="stub")
    model = FakeModel()
    with pytest.raises(ValueError, match="no-such-model"):
        submit(connection, model, run_id, [LORA.arxiv_id])
    assert model.calls == []


class Explodes(FakeModel):
    def count_tokens(self, requests, *, model):
        raise RuntimeError("network down")


@pytest.mark.parametrize(
    "model, error",
    [
        (FakeModel(counted_tokens=30_000_000), BudgetExceeded),
        (FakeModel(fail_submit=True), RuntimeError),
        (Explodes(), RuntimeError),
    ],
)
def test_failed_submit_leaves_no_open_transaction_and_no_lock(connection, model, error):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    with pytest.raises(error):
        submit(connection, model, run_id, [LORA.arxiv_id], ceiling_usd=Decimal(1))
    assert connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    assert advisory_locks(connection) == 0


def test_submit_and_collect_refuse_a_connection_with_an_open_transaction(connection):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    batch_id = submit(connection, FakeModel(), run_id, [LORA.arxiv_id])
    model = FakeModel(LORA_ANSWER)
    with connection.transaction():
        connection.execute("UPDATE extraction_run SET note = 'callers work'")
        with pytest.raises(ValueError, match="transaction"):
            submit(connection, model, run_id, [SURVEY.arxiv_id])
        with pytest.raises(ValueError, match="transaction"):
            collect(connection, model, batch_id)
    assert model.calls == []
    # Neither call rolled the caller's work back.
    assert connection.execute("SELECT note FROM extraction_run").fetchone() == ("callers work",)


def test_collected_result_of_an_outdated_abstract_does_not_overwrite_a_newer_one(connection):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    model = FakeModel()
    old_batch = submit(connection, model, run_id, [LORA.arxiv_id])
    edited = ExtractionRequest(LORA.arxiv_id, LORA.title, "We study LoRA for summarization.")
    connection.execute(
        "UPDATE paper SET abstract = %s, abstract_hash = %s",
        (edited.abstract, abstract_hash(edited.abstract)),
    )
    new_batch = submit(connection, model, run_id, [edited.arxiv_id])
    newer = ExtractionResult(
        LORA.arxiv_id,
        PaperMentions(mentions=[Mention(entity_type="task", spelling="summarization")]),
        50,
        10,
    )
    model.batch_answers = {
        "batch-1": {LORA.arxiv_id: LORA_ANSWER},
        "batch-2": {LORA.arxiv_id: newer},
    }
    assert collect(connection, model, new_batch) == Counter(extracted=1, mentions=1)
    assert collect(connection, model, old_batch) == Counter(stale=1)
    assert mention_rows(connection) == [
        ("2511.00001", "task", "summarization", "summarization", None)
    ]
    assert connection.execute(
        "SELECT abstract_hash, input_tokens FROM paper_extraction"
    ).fetchall() == [(abstract_hash(edited.abstract), 50)]
    # The outdated answer was billed all the same.
    assert spent_usd(connection) == usd(170, 50)


def test_collected_batch_records_what_it_cost_including_unusable_answers(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel(ExtractionResult(LORA.arxiv_id, None, 120, 40), SURVEY_ANSWER)
    batch_id, _ = run_batch(connection, model, new_run(connection), LORA, SURVEY)
    assert connection.execute(
        "SELECT actual_cost_usd FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone() == (usd(210, 45),)


def test_spent_money_survives_deleting_papers_and_a_run_with_batches_cannot_be_deleted(
    connection,
):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    run_batch(connection, FakeModel(LORA_ANSWER), run_id, LORA)
    spent = spent_usd(connection)
    connection.execute("DELETE FROM paper")
    assert spent_usd(connection) == spent > 0
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.execute("DELETE FROM extraction_run WHERE id = %s", (run_id,))


def test_blank_spelling_is_rejected():
    for spelling in ("", "   "):
        with pytest.raises(ValidationError):
            Mention(entity_type="task", spelling=spelling)


def test_mention_with_an_empty_normal_form_is_not_stored(connection):
    add_papers(connection, LORA)
    answer = ExtractionResult(
        LORA.arxiv_id,
        PaperMentions(
            mentions=[
                Mention(entity_type="task", spelling="-"),
                Mention(entity_type="method", spelling="LoRA"),
            ]
        ),
        10,
        5,
    )
    _, counter = run_batch(connection, FakeModel(answer), new_run(connection), LORA)
    assert counter == Counter(extracted=1, mentions=1)
    assert [row[2] for row in mention_rows(connection)] == ["LoRA"]


class FakeSdk:
    """The SDK client as far as counting goes: 100 tokens of overhead and a token a char."""

    def __init__(self):
        self.messages = self
        self.counted = []

    def count_tokens(self, **params):
        text = params["messages"][0]["content"]
        self.counted.append(text)
        return SimpleNamespace(input_tokens=100 + len(text))


def test_real_client_counts_overhead_once_per_request_and_abstracts_in_glued_parts(monkeypatch):
    monkeypatch.setattr(extraction, "COUNT_CHARS_PER_CALL", 200)
    sdk = FakeSdk()
    client = extraction.AnthropicModelClient.__new__(extraction.AnthropicModelClient)
    client._client = sdk
    requests = [ExtractionRequest(f"id{i}", f"t{i}", "x" * 60) for i in range(7)]
    total = client.count_tokens(requests, model="the-model")
    overhead_text, *parts = sdk.counted
    assert "x" not in overhead_text
    # Far fewer calls than requests, and every paper is in exactly one part.
    assert 1 < len(parts) < len(requests)
    assert all(sum(part.count(f"Title: t{i}") for part in parts) == 1 for i in range(7))
    overhead = 100 + len(overhead_text)
    assert total == sum(100 + len(part) for part in parts) + len(requests) * overhead


def test_counting_happens_before_the_budget_lock_is_taken(connection):
    add_papers(connection, LORA)

    class Probing(FakeModel):
        locks_while_counting = None

        def count_tokens(self, requests, *, model):
            self.locks_while_counting = advisory_locks(connection)
            return super().count_tokens(requests, model=model)

    model = Probing()
    submit(connection, model, new_run(connection), [LORA.arxiv_id])
    assert model.locks_while_counting == 0


def test_batch_refused_outright_is_not_money_spent_and_frees_its_papers(connection):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    with pytest.raises(BatchRejected):
        submit(connection, FakeModel(reject_submit=True), run_id, [LORA.arxiv_id])
    assert connection.execute("SELECT status FROM extraction_batch").fetchone() == ("rejected",)
    assert spent_usd(connection) == 0
    # The papers are due again.
    healthy = FakeModel()
    assert submit(connection, healthy, run_id, [LORA.arxiv_id]) is not None
    assert healthy.sent[0][0].arxiv_id == LORA.arxiv_id


def test_run_model_is_what_the_client_is_called_with_and_priced_by(connection, monkeypatch):
    monkeypatch.setitem(
        extraction.BATCH_PRICES_USD_PER_MTOK, "other-model", (Decimal(1), Decimal(2))
    )
    add_papers(connection, LORA)
    run_id = start_run(connection, model="other-model", prompt_hash="stub")
    model = FakeModel(counted_tokens=1_000_000)
    batch_id = submit(connection, model, run_id, [LORA.arxiv_id])
    assert model.models == ["other-model", "other-model"]
    estimate = connection.execute(
        "SELECT estimated_cost_usd FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone()[0]
    assert estimate == extraction._usd("other-model", 1_000_000, EXTRACTION_MAX_OUTPUT_TOKENS)


def test_submit_sends_the_text_and_hash_stored_in_the_database(connection):
    add_papers(connection, LORA)
    connection.execute(
        "UPDATE paper SET title = 'Stored title', abstract = 'Stored abstract.',"
        " abstract_hash = %s",
        (abstract_hash("Stored abstract."),),
    )
    model = FakeModel()
    batch_id = submit(connection, model, new_run(connection), [LORA.arxiv_id])
    assert model.sent == [[ExtractionRequest(LORA.arxiv_id, "Stored title", "Stored abstract.")]]
    assert connection.execute(
        "SELECT abstract_hash FROM extraction_batch_paper WHERE batch_id = %s", (batch_id,)
    ).fetchone() == (abstract_hash("Stored abstract."),)


def test_repeated_ids_are_sent_once_in_the_order_given(connection):
    add_papers(connection, LORA, SURVEY)
    model = FakeModel()
    submit(
        connection,
        model,
        new_run(connection),
        [SURVEY.arxiv_id, LORA.arxiv_id, SURVEY.arxiv_id],
    )
    assert [r.arxiv_id for r in model.sent[0]] == [SURVEY.arxiv_id, LORA.arxiv_id]


def test_collect_reads_the_network_outside_a_transaction_and_rechecks_the_batch(connection):
    add_papers(connection, LORA)
    run_id = new_run(connection)
    model = FakeModel(LORA_ANSWER)
    batch_id = submit(connection, model, run_id, [LORA.arxiv_id])
    seen_status = []
    rival = FakeModel(LORA_ANSWER)
    rival.batches = model.batches

    class Slow(FakeModel):
        def batch_ended(self, provider_id):
            seen_status.append(connection.info.transaction_status)
            return True

        def batch_results(self, provider_id):
            results = list(super().batch_results(provider_id))
            seen_status.append(connection.info.transaction_status)
            # While this collector waited on the network, another one stored the batch.
            assert collect(connection, rival, batch_id) == Counter(extracted=1, mentions=4)
            return iter(results)

    slow = Slow(LORA_ANSWER)
    slow.batches = model.batches
    assert collect(connection, slow, batch_id) == Counter()
    assert seen_status == [psycopg.pq.TransactionStatus.IDLE] * 2
    assert len(mention_rows(connection)) == 4


def test_collect_holds_a_share_lock_on_the_papers_it_stores_against(connection, postgres_url):
    add_papers(connection, LORA)
    model = FakeModel(LORA_ANSWER)
    batch_id = submit(connection, model, new_run(connection), [LORA.arxiv_id])
    outcome = []
    real_store = extraction._store

    def store_while_another_session_edits(*args, **kwargs):
        with psycopg.connect(postgres_url, autocommit=True) as other:
            other.execute("SET lock_timeout = '200ms'")
            try:
                other.execute(
                    "UPDATE paper SET abstract = 'edited' WHERE arxiv_id = %s", (LORA.arxiv_id,)
                )
                outcome.append("edited")
            except psycopg.errors.LockNotAvailable:
                outcome.append("blocked")
        return real_store(*args, **kwargs)

    extraction._store = store_while_another_session_edits
    try:
        collect(connection, model, batch_id)
    finally:
        extraction._store = real_store
    assert outcome == ["blocked"]


class FakeBatches:
    def __init__(self, error=None):
        self.error = error
        self.created = []

    def create(self, *, requests):
        if self.error:
            raise self.error
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_1")


class FakeSdkWithBatches:
    def __init__(self, error=None):
        self.options = []
        self.batches = FakeBatches(error)
        self.messages = SimpleNamespace(batches=self.batches)

    def with_options(self, **options):
        self.options.append(options)
        return self


def real_client(sdk):
    client = extraction.AnthropicModelClient.__new__(extraction.AnthropicModelClient)
    client._client = sdk
    return client


def test_real_client_creates_the_batch_once_with_encoded_ids_and_the_runs_model():
    sdk = FakeSdkWithBatches()
    requests = [LORA, ExtractionRequest("2511.12345", "t", "a")]
    assert real_client(sdk).submit_batch(requests, model="the-model") == "msgbatch_1"
    # A retry by the SDK could create a second, paid batch: no request id protects it.
    assert sdk.options == [{"max_retries": 0}]
    entries = sdk.batches.created[0]
    assert [e["custom_id"] for e in entries] == ["2511_00001", "2511_12345"]
    for entry in entries:
        assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", entry["custom_id"])
        assert entry["params"]["model"] == "the-model"
        assert entry["params"]["max_tokens"] == EXTRACTION_MAX_OUTPUT_TOKENS


def test_batch_entry_custom_id_is_decoded_back_to_the_arxiv_id():
    entry = batch_entry("succeeded", '{"mentions": []}')
    entry.custom_id = "2511_00001"
    assert extraction.result_from_batch_entry(entry).arxiv_id == "2511.00001"


def test_real_client_refuses_an_id_it_cannot_encode_before_calling_the_api():
    sdk = FakeSdkWithBatches()
    with pytest.raises(BatchRejected):
        real_client(sdk).submit_batch([ExtractionRequest("hep-th/9901001", "t", "a")], model="m")
    assert sdk.batches.created == []


@pytest.mark.parametrize(
    "name", ["BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError"]
)
def test_real_client_turns_a_4xx_refusal_into_batch_rejected(name):
    anthropic = pytest.importorskip("anthropic")
    error = getattr(anthropic, name).__new__(getattr(anthropic, name))
    with pytest.raises(BatchRejected):
        real_client(FakeSdkWithBatches(error)).submit_batch([LORA], model="m")


@pytest.mark.parametrize("name", ["InternalServerError", "APIConnectionError", "APITimeoutError"])
def test_real_client_lets_an_unclear_failure_through_unchanged(name):
    anthropic = pytest.importorskip("anthropic")
    error = getattr(anthropic, name).__new__(getattr(anthropic, name))
    with pytest.raises(type(error)):
        real_client(FakeSdkWithBatches(error)).submit_batch([LORA], model="m")
