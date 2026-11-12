from types import SimpleNamespace

import psycopg
import pytest

from research_radar import extraction
from research_radar.extraction import (
    DEFAULT_CONFIG,
    ExtractionRequest,
    RequestConfig,
    RunConfigMismatch,
    config_hash,
    request_params,
    start_run,
    submit,
)
from research_radar.papers import abstract_hash
from research_radar.settings import EXTRACTION_MAX_OUTPUT_TOKENS, EXTRACTION_MODEL

LORA = ExtractionRequest("2511.00001", "Adapting a translator", "We fine-tune Llama with LoRA.")
NO_REASONING = RequestConfig(reasoning=False, max_output_tokens=512)


class FakeModel:
    """Stands in for the model client; logs the calls and the configuration each one got."""

    def __init__(self, counted_tokens=1000):
        self.counted_tokens = counted_tokens
        self.calls = []
        self.configs = []

    def count_tokens(self, requests, *, model, config):
        self.calls.append("count_tokens")
        self.configs.append(config)
        return self.counted_tokens

    def submit_batch(self, requests, *, model, config):
        self.calls.append("submit_batch")
        self.configs.append(config)
        return "batch-1"


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            """
            INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                               primary_category, first_version_date, arxiv_updated_at)
            VALUES (%s, 1, %s, %s, %s, '[]', 'cs.CL', '2025-11-03', '2025-11-03')
            """,
            (LORA.arxiv_id, LORA.title, LORA.abstract, abstract_hash(LORA.abstract)),
        )
        yield connection


def run_row(connection, run_id):
    return connection.execute(
        "SELECT prompt_hash, request_config FROM extraction_run WHERE id = %s", (run_id,)
    ).fetchone()


def batch_count(connection):
    return connection.execute("SELECT count(*) FROM extraction_batch").fetchone()[0]


def test_run_records_the_hash_of_its_request_config(connection):
    default = start_run(connection, model=EXTRACTION_MODEL)
    assert run_row(connection, default) == (
        config_hash(DEFAULT_CONFIG),
        {"reasoning": True, "max_output_tokens": EXTRACTION_MAX_OUTPUT_TOKENS},
    )
    own = start_run(connection, model=EXTRACTION_MODEL, config=NO_REASONING)
    assert run_row(connection, own) == (
        config_hash(NO_REASONING),
        {"reasoning": False, "max_output_tokens": 512},
    )
    assert config_hash(NO_REASONING) != config_hash(DEFAULT_CONFIG)


def test_changed_system_prompt_stops_submit_to_an_old_run(connection, monkeypatch):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    monkeypatch.setattr(extraction, "EXTRACTION_PROMPT", "A prompt edited after the run began.")
    client = FakeModel()
    with pytest.raises(RunConfigMismatch):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []
    assert batch_count(connection) == 0


def test_changed_message_template_changes_the_hash(monkeypatch):
    before = config_hash(DEFAULT_CONFIG)
    monkeypatch.setattr(extraction, "MESSAGE_TEMPLATE", "Paper: {title}\n\nText: {abstract}")
    assert config_hash(DEFAULT_CONFIG) != before


def test_changed_output_schema_changes_the_hash(monkeypatch):
    before = config_hash(DEFAULT_CONFIG)
    monkeypatch.setattr(extraction, "_output_schema", lambda: {"type": "object"})
    assert config_hash(DEFAULT_CONFIG) != before


def test_hash_does_not_depend_on_the_order_of_dictionary_keys():
    one = extraction._canonical_json({"b": 1, "a": {"y": 1, "x": 2}})
    other = extraction._canonical_json({"a": {"x": 2, "y": 1}, "b": 1})
    assert one == other


def test_reasoning_off_and_on_are_different_configs_and_requests():
    on = request_params(LORA, "some-model", RequestConfig(reasoning=True))
    off = request_params(LORA, "some-model", RequestConfig(reasoning=False))
    # With reasoning, the request is what it was before the config existed: effort low and
    # no thinking parameter. Haiku 5.5 thinks by default; "disabled" is accepted at effort
    # high or below (claude-api skill), and this request is at low.
    assert "thinking" not in on
    assert on["output_config"]["effort"] == "low"
    assert off == {**on, "thinking": {"type": "disabled"}}
    assert config_hash(RequestConfig(reasoning=True)) != config_hash(RequestConfig(reasoning=False))


def test_run_without_reasoning_is_submitted_with_its_own_config(connection):
    run_id = start_run(connection, model=EXTRACTION_MODEL, config=NO_REASONING)
    client = FakeModel()
    assert submit(connection, client, run_id, [LORA.arxiv_id]) is not None
    assert client.calls == ["count_tokens", "submit_batch"]
    assert client.configs == [NO_REASONING, NO_REASONING]


def test_estimate_uses_the_runs_max_output_tokens(connection):
    run_id = start_run(connection, model=EXTRACTION_MODEL, config=NO_REASONING)
    batch_id = submit(connection, FakeModel(counted_tokens=1_000_000), run_id, [LORA.arxiv_id])
    stored = connection.execute(
        "SELECT estimated_cost_usd FROM extraction_batch WHERE id = %s", (batch_id,)
    ).fetchone()[0]
    assert stored == extraction._usd(EXTRACTION_MODEL, 1_000_000, 512)


def test_run_from_before_the_migration_counts_as_the_default_config(connection):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    connection.execute("UPDATE extraction_run SET request_config = '{}' WHERE id = %s", (run_id,))
    client = FakeModel()
    assert submit(connection, client, run_id, [LORA.arxiv_id]) is not None
    assert client.configs == [DEFAULT_CONFIG, DEFAULT_CONFIG]


def test_old_run_with_another_hash_is_refused_and_left_as_it_was(connection):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    connection.execute(
        "UPDATE extraction_run SET request_config = '{}', prompt_hash = 'old-prompt-only'"
        " WHERE id = %s",
        (run_id,),
    )
    client = FakeModel()
    with pytest.raises(RunConfigMismatch):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []
    assert run_row(connection, run_id) == ("old-prompt-only", {})
    assert batch_count(connection) == 0


class FakeBatches:
    def __init__(self):
        self.created = []

    def create(self, *, requests):
        self.created.append(requests)
        return SimpleNamespace(id="msgbatch_1")


def test_real_client_sends_the_runs_config_in_every_batch_entry():
    batches = FakeBatches()
    sdk = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    sdk.with_options = lambda **options: sdk
    client = extraction.AnthropicModelClient.__new__(extraction.AnthropicModelClient)
    client._client = sdk
    client.submit_batch([LORA], model="m", config=NO_REASONING)
    (entry,) = batches.created[0]
    assert entry["params"]["max_tokens"] == 512
    assert entry["params"]["thinking"] == {"type": "disabled"}
