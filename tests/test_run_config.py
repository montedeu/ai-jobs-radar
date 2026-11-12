import psycopg
import pytest
from fakes import FakeModel, FakeSdkWithBatches, real_client

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
from research_radar.papers import abstract_hash, text_hash
from research_radar.settings import EXTRACTION_MAX_OUTPUT_TOKENS, EXTRACTION_MODEL

LORA = ExtractionRequest("2511.00001", "Adapting a translator", "We fine-tune Llama with LoRA.")
NO_REASONING = RequestConfig(reasoning=False, max_output_tokens=512)


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


def hash_of(config):
    return config_hash(config, EXTRACTION_MODEL)


def test_run_records_the_hash_of_its_request_config(connection):
    default = start_run(connection, model=EXTRACTION_MODEL)
    assert run_row(connection, default) == (
        hash_of(DEFAULT_CONFIG),
        {"reasoning": True, "max_output_tokens": EXTRACTION_MAX_OUTPUT_TOKENS},
    )
    own = start_run(connection, model=EXTRACTION_MODEL, config=NO_REASONING)
    assert run_row(connection, own) == (
        hash_of(NO_REASONING),
        {"reasoning": False, "max_output_tokens": 512},
    )
    assert hash_of(NO_REASONING) != hash_of(DEFAULT_CONFIG)


def test_changed_system_prompt_stops_submit_to_an_old_run(connection, monkeypatch):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    monkeypatch.setattr(extraction, "EXTRACTION_PROMPT", "A prompt edited after the run began.")
    client = FakeModel()
    with pytest.raises(RunConfigMismatch):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []
    assert batch_count(connection) == 0


def test_changed_message_template_changes_the_hash(monkeypatch):
    before = hash_of(DEFAULT_CONFIG)
    monkeypatch.setattr(extraction, "MESSAGE_TEMPLATE", "Paper: {title}\n\nText: {abstract}")
    assert hash_of(DEFAULT_CONFIG) != before


def test_changed_output_schema_changes_the_hash(monkeypatch):
    before = hash_of(DEFAULT_CONFIG)
    monkeypatch.setattr(extraction, "_output_schema", lambda: {"type": "object"})
    assert hash_of(DEFAULT_CONFIG) != before


def test_changed_effort_in_the_request_stops_submit_to_an_old_run(connection, monkeypatch):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    built = request_params

    def with_high_effort(*args, **kwargs):
        params = built(*args, **kwargs)
        params["output_config"] = {**params["output_config"], "effort": "high"}
        return params

    monkeypatch.setattr(extraction, "request_params", with_high_effort)
    assert hash_of(DEFAULT_CONFIG) != run_row(connection, run_id)[0]
    client = FakeModel()
    with pytest.raises(RunConfigMismatch):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []


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
    assert hash_of(RequestConfig(reasoning=True)) != hash_of(RequestConfig(reasoning=False))


def test_reasoning_is_refused_for_a_model_not_known_to_think_by_default(connection):
    # Without a thinking parameter the request relies on the model's default.
    with pytest.raises(ValueError, match="other-model"):
        start_run(connection, model="other-model", config=RequestConfig(reasoning=True))
    assert connection.execute("SELECT count(*) FROM extraction_run").fetchone()[0] == 0
    start_run(connection, model="other-model", config=RequestConfig(reasoning=False))


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
    assert stored == extraction.estimate_usd(EXTRACTION_MODEL, 1_000_000, 512)


def test_run_from_before_the_migration_is_refused_and_left_as_it_was(connection):
    # A real old row: the hash of the prompt text alone, and the column's default.
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    old_hash = text_hash(extraction.EXTRACTION_PROMPT)
    connection.execute(
        "UPDATE extraction_run SET request_config = '{}', prompt_hash = %s WHERE id = %s",
        (old_hash, run_id),
    )
    client = FakeModel()
    with pytest.raises(RunConfigMismatch, match="request configuration"):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []
    assert run_row(connection, run_id) == (old_hash, {})
    assert batch_count(connection) == 0


@pytest.mark.parametrize(
    "stored",
    [
        '{"reasoning": true, "max_output_tokens": "512"}',
        '{"reasoning": true, "max_output_tokens": 0}',
        '{"reasoning": true, "max_output_tokens": 2048, "extra": 1}',
        '{"reasoning": "yes", "max_output_tokens": 2048}',
        '{"reasoning": true}',
    ],
)
def test_stored_config_of_the_wrong_shape_is_refused(connection, stored):
    run_id = start_run(connection, model=EXTRACTION_MODEL)
    connection.execute(
        "UPDATE extraction_run SET request_config = %s WHERE id = %s", (stored, run_id)
    )
    client = FakeModel()
    with pytest.raises(RunConfigMismatch):
        submit(connection, client, run_id, [LORA.arxiv_id])
    assert client.calls == []


def test_real_client_sends_the_runs_config_in_every_batch_entry():
    sdk = FakeSdkWithBatches()
    real_client(sdk).submit_batch([LORA], model="m", config=NO_REASONING)
    (entry,) = sdk.batches.created[0]
    assert entry["params"]["max_tokens"] == 512
    assert entry["params"]["thinking"] == {"type": "disabled"}
