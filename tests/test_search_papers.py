from datetime import UTC, date, datetime, timedelta

import psycopg
import pytest
from starlette.testclient import TestClient

from research_radar import mcp_server, search
from research_radar.extraction import start_run
from research_radar.normalize import normal_form
from research_radar.runs import promote
from research_radar.settings import EXTRACTION_MAX_ATTEMPTS, EXTRACTION_MODEL

RAG = "Retrieval-Augmented Generation"
TOKEN = "test-token"
MCP_HEADERS = {"Accept": "application/json, text/event-stream"}


@pytest.fixture
def client(database_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", database_url)
    with TestClient(mcp_server.application(TOKEN)) as client:
        yield client


def call(client, method, params=None, token=TOKEN):
    headers = dict(MCP_HEADERS)
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    return client.post("/mcp", headers=headers, json=body)


@pytest.fixture
def connection(database_url):
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


def add_paper(connection, arxiv_id, title, abstract, day, categories=("cs.CL",)):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                           primary_category, first_version_date, arxiv_updated_at)
        VALUES (%s, 1, %s, %s, 'h', '[]', %s, %s, '2025-12-01')
        """,
        (arxiv_id, title, abstract, categories[0], day),
    )
    for category in categories:
        connection.execute(
            "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, %s)", (arxiv_id, category)
        )


def mark_done(connection, run_id, arxiv_id, *spellings, entity_type="method"):
    connection.execute(
        "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
        " output_tokens) VALUES (%s, %s, 'h', 1, 1)",
        (run_id, arxiv_id),
    )
    for spelling in spellings:
        connection.execute(
            "INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form)"
            " VALUES (%s, %s, %s, %s, %s)",
            (run_id, arxiv_id, entity_type, spelling, normal_form(spelling)),
        )


def set_collected_through(connection, category, day):
    connection.execute(
        "INSERT INTO collection_state (category, backfilled_through) VALUES (%s, %s)"
        " ON CONFLICT (category) DO UPDATE SET backfilled_through = excluded.backfilled_through",
        (category, day),
    )


@pytest.fixture
def world(connection):
    """Run 1 is current. 2511.00001 and 2511.00002 are in it; 2512.00003 came after the
    run started and has no completion mark."""
    add_paper(
        connection,
        "2511.00001",
        "Retrieval-augmented generation for QA",
        "We ground answers in retrieved passages.",
        "2025-11-03",
    )
    add_paper(
        connection, "2511.00002", "LoRA for MT", "Low rank adapters for translation.", "2025-11-10"
    )
    add_paper(
        connection, "2511.00004", "Vision models", "Retrieval of images.", "2025-11-05", ("cs.CV",)
    )
    run = start_run(connection, model=EXTRACTION_MODEL, full=True)
    mark_done(connection, run, "2511.00001", "RAG")
    mark_done(connection, run, "2511.00002", "LoRA")
    assert promote(connection) == run
    add_paper(
        connection, "2512.00003", "A survey", "Many methods, retrieval included.", "2025-12-01"
    )
    connection.execute(
        "INSERT INTO synonym VALUES ('rag', %s), ('retrieval augmented generation', %s)",
        (RAG, RAG),
    )
    connection.execute("INSERT INTO synonym_version (file_sha256, entries) VALUES ('abc', 2)")
    set_collected_through(connection, "cs.CL", date(2025, 12, 31))
    return run


def search_tool(client, **arguments):
    response = call(client, "tools/call", {"name": "search_papers", "arguments": arguments})
    assert response.status_code == 200
    return response.json()["result"]


def found(client, **arguments):
    result = search_tool(client, **arguments)
    assert not result.get("isError"), result
    return result["structuredContent"]


def ids(answer):
    return [paper["arxiv_id"] for paper in answer["papers"]]


def test_text_search_finds_papers_by_title_and_abstract(client, world):
    assert ids(found(client, query="LoRA")) == ["2511.00002"]
    assert ids(found(client, query="adapters")) == ["2511.00002"]
    assert set(ids(found(client, query="retrieval"))) == {"2512.00003", "2511.00001"}
    (hit,) = found(client, query="adapters")["papers"]
    assert hit == {
        "arxiv_id": "2511.00002",
        "title": "LoRA for MT",
        "date": "2025-11-10",
        "url": "https://arxiv.org/abs/2511.00002",
    }


def test_text_search_uses_the_index_expression(connection, world):
    connection.execute("SET enable_seqscan = off")
    sql, params = search.build_query(
        query="retrieval", entity=None, category="cs.CL", date_from=None, date_to=None, limit=20
    )
    plan = "\n".join(row[0] for row in connection.execute("EXPLAIN " + sql, params).fetchall())
    assert "paper_search" in plan


def test_filters_by_slice_and_date_range(client, world):
    assert ids(found(client, query="retrieval", category="cs.CL")) != ["2511.00004"]
    assert ids(found(client, date_from="2025-11-05", date_to="2025-11-30")) == ["2511.00002"]
    assert ids(found(client, date_from="2025-11-10")) == ["2512.00003", "2511.00002"]
    assert ids(found(client, date_to="2025-11-03")) == ["2511.00001"]
    assert "2511.00004" not in ids(found(client))


def test_papers_without_a_filter_come_newest_first(client, world):
    assert ids(found(client)) == ["2512.00003", "2511.00002", "2511.00001"]
    assert ids(found(client, limit=2)) == ["2512.00003", "2511.00002"]


def test_entity_in_any_spelling_finds_papers_of_the_current_run(client, world):
    for spelling in [RAG, "retrieval augmented generation", "RAG", "rags"]:
        assert ids(found(client, entity=spelling)) == ["2511.00001"], spelling
    assert ids(found(client, entity="lora")) == ["2511.00002"]


def test_entity_search_ignores_other_runs(client, connection, world):
    other = start_run(connection, model=EXTRACTION_MODEL)
    mark_done(connection, other, "2511.00001", "LoRA")

    assert ids(found(client, entity="LoRA")) == ["2511.00002"]


def test_text_and_entity_together(client, world):
    assert ids(found(client, query="generation", entity="rag")) == ["2511.00001"]
    answer = found(client, query="generation", entity="LoRA")
    assert answer["papers"] == []
    assert answer["message"]


def test_slice_that_is_not_collected_gets_a_message(client, world):
    answer = found(client, category="cs.CV")
    assert answer["papers"] == []
    assert "cs.CL" in answer["message"]


def test_entity_search_without_current_run_explains_why(client, connection):
    add_paper(connection, "2511.00001", "RAG", "RAG.", "2025-11-03")

    answer = found(client, entity="RAG")

    assert answer["papers"] == []
    assert "run" in answer["message"]
    assert answer["header"]["extraction_run"] is None
    assert answer["header"]["processed_through"] is None


def test_unknown_entity_and_stop_words_explain_the_empty_answer(client, world):
    unknown = found(client, entity="Quantum Gravity Transformer")
    assert unknown["papers"] == []
    assert "Quantum Gravity Transformer" in unknown["message"]
    stops = found(client, query="the of and")
    assert stops["papers"] == []
    assert stops["message"]


def test_a_found_answer_has_no_message(client, world):
    assert found(client, query="adapters")["message"] is None


def test_date_range_the_wrong_way_round_is_an_error_with_a_reason(client, world):
    result = search_tool(client, date_from="2025-12-01", date_to="2025-11-01")
    assert result["isError"]
    assert "date_from" in result["content"][0]["text"]


@pytest.mark.parametrize("limit", [0, 101])
def test_limit_outside_one_to_a_hundred_is_an_error(client, world, limit):
    assert search_tool(client, limit=limit)["isError"]


def test_response_has_the_common_header(client, world):
    before = datetime.now(UTC).date() - timedelta(days=1)
    header = found(client, query="adapters")["header"]
    after = datetime.now(UTC).date() - timedelta(days=1)

    assert header["as_of"] in {before.isoformat(), after.isoformat()}
    assert header["extraction_run"] == world
    assert header["processed_through"] == "2025-11-30"
    assert header["dictionary_updated_at"] is not None
    assert header["dictionary_sha256"] == "abc"
    notes = " ".join(header["notes"])
    assert len(header["notes"]) == 4
    for fragment in ["чаще упоминают", "Иерархии", "cs.CL", "arXiv"]:
        assert fragment in notes


def test_header_without_a_dictionary_has_no_dictionary_fields(client, connection, world):
    connection.execute("DELETE FROM synonym_version")

    header = found(client, query="adapters")["header"]

    assert header["dictionary_updated_at"] is None
    assert header["dictionary_sha256"] is None


def test_processed_through_stops_before_the_first_unprocessed_paper(client, connection, world):
    add_paper(connection, "2512.00009", "Later", "Later.", "2025-12-20")
    mark_done(connection, world, "2512.00009")

    assert found(client)["header"]["processed_through"] == "2025-11-30"


def test_processed_through_is_the_last_marked_day_when_nothing_is_missing(
    client, connection, world
):
    connection.execute("DELETE FROM paper WHERE arxiv_id = '2512.00003'")

    assert found(client)["header"]["processed_through"] == "2025-11-10"


def test_processed_through_skips_papers_the_run_gave_up_on(client, connection, world):
    add_paper(connection, "2512.00009", "Later", "Later.", "2025-12-20")
    mark_done(connection, world, "2512.00009")
    for _ in range(EXTRACTION_MAX_ATTEMPTS):
        batch = connection.execute(
            "INSERT INTO extraction_batch (run_id, status, estimated_cost_usd)"
            " VALUES (%s, 'collected', 0) RETURNING id",
            (world,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO extraction_batch_paper (batch_id, arxiv_id, abstract_hash, outcome)"
            " VALUES (%s, '2512.00003', 'h', 'invalid')",
            (batch,),
        )

    assert found(client)["header"]["processed_through"] == "2025-12-20"


def test_processed_through_is_capped_by_collected_history(client, connection, world):
    set_collected_through(connection, "cs.CL", date(2025, 11, 20))

    assert found(client)["header"]["processed_through"] == "2025-11-20"


def test_processed_through_is_none_when_a_slice_has_no_collected_history(client, connection, world):
    connection.execute("DELETE FROM collection_state")

    assert found(client)["header"]["processed_through"] is None


def test_tool_does_not_touch_arxiv(client, world):
    # The autouse fixture no_real_arxiv fails any request to arXiv; an ordinary call passes.
    assert found(client, query="retrieval", entity="rag")["papers"]


def test_search_without_token_is_rejected(client, world):
    response = call(client, "tools/call", {"name": "search_papers", "arguments": {}}, token=None)
    assert response.status_code == 401
