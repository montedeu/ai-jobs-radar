from datetime import UTC, datetime

import psycopg
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from research_radar import collect, history_summary, mcp_server
from research_radar.database import connect

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


def test_server_refuses_to_start_without_a_token():
    with pytest.raises(SystemExit):
        mcp_server.application("")


def test_health_needs_no_token(client):
    assert client.get("/health").status_code == 200


def test_health_fails_when_the_database_cannot_be_read(client, monkeypatch):
    def unreachable():
        raise ConnectionError("no route to the database")

    monkeypatch.setattr(mcp_server, "connect", unreachable)

    response = client.get("/health")

    assert response.status_code == 503
    assert "ConnectionError" in response.text


def test_health_fails_cleanly_when_no_database_is_configured(client, monkeypatch):
    monkeypatch.delenv("DATABASE_URL")

    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("path", ["/health/", "//health", "/health/../mcp", "/HEALTH"])
def test_only_the_exact_health_path_is_open(client, path):
    assert client.get(path).status_code == 401


def test_websocket_is_refused(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/mcp"):
        pass


@pytest.mark.parametrize("token", [None, "wrong", TOKEN + "x", ""])
def test_mcp_endpoint_rejects_a_missing_or_wrong_token(client, token):
    assert call(client, "tools/list", token=token).status_code == 401


def test_unknown_paths_are_rejected_before_routing(client):
    assert client.get("/anything").status_code == 401


def test_service_status_reports_the_latest_collect_run(client, database_url, fake_arxiv):
    collect.run(database_url, fake_arxiv)
    response = call(client, "tools/call", {"name": "service_status", "arguments": {}})
    assert response.status_code == 200
    (run,) = response.json()["result"]["structuredContent"]["latest_runs"]
    assert (run["job"], run["status"]) == ("collect", "succeeded")


def status(client):
    response = call(client, "tools/call", {"name": "service_status", "arguments": {}})
    assert response.status_code == 200
    return response.json()["result"]["structuredContent"]["history"]


def add_paper(connection, arxiv_id, day, categories):
    connection.execute(
        """
        INSERT INTO paper (arxiv_id, title, abstract, abstract_hash, authors, primary_category,
                           first_version_date, version, arxiv_updated_at)
        VALUES (%s, 't', 'a', 'h', '[]', %s, %s, 1, now())
        """,
        (arxiv_id, categories[0], day),
    )
    for category in categories:
        connection.execute(
            "INSERT INTO paper_category (arxiv_id, category) VALUES (%s, %s)", (arxiv_id, category)
        )


def set_state(connection, category, backfilled_through, updated_through):
    connection.execute(
        "INSERT INTO collection_state (category, backfilled_through, updated_through)"
        " VALUES (%s, %s, %s)",
        (category, backfilled_through, updated_through),
    )


@pytest.fixture
def collected(database_url):
    """cs.CL collected through 2025-12-03; one cs.CL paper is primary cs.LG."""
    with connect(database_url) as connection:
        set_state(connection, "cs.CL", "2025-12-03", "2025-12-03T18:00Z")
        days = ["2025-11-01", "2025-11-01", "2025-11-03", "2025-12-01", "2025-12-03", "2025-12-05"]
        for number, day in enumerate(days):
            categories = ["cs.LG", "cs.CL"] if day == "2025-11-03" else ["cs.CL"]
            add_paper(connection, f"2511.{number:05d}", day, categories)
        connection.commit()
    return database_url


def cs_cl(history):
    (found,) = [s for s in history["slices"] if s["category"] == "cs.CL"]
    return found


def test_service_status_counts_papers_per_month_for_each_slice(client, collected):
    history = status(client)

    assert [(m["month"], m["papers"]) for m in cs_cl(history)["months"]] == [
        ("2025-11", 3),
        ("2025-12", 3),
    ]


def test_paper_with_cs_cl_as_secondary_category_is_counted_in_cs_cl(client, database_url):
    with connect(database_url) as connection:
        add_paper(connection, "2511.00001", "2025-11-03", ["cs.LG", "cs.CL"])
        connection.commit()

    assert [(m["month"], m["papers"]) for m in cs_cl(status(client))["months"]] == [("2025-11", 1)]


def test_month_not_fully_collected_is_marked_partial(client, collected):
    months = cs_cl(status(client))["months"]

    assert [(m["month"], m["partial"]) for m in months] == [("2025-11", False), ("2025-12", True)]


def test_service_status_reports_backfill_boundary_and_last_daily_collection(
    client, collected, database_url
):
    with connect(database_url) as connection:
        for outcome, finished in [
            ("succeeded", "2025-12-05T06:00Z"),
            ("skipped", "2025-12-06T06:00Z"),
            ("failed", "2025-12-07T06:00Z"),
        ]:
            connection.execute(
                "INSERT INTO job_run (job, status, started_at, finished_at)"
                " VALUES ('collect', %s, %s, %s)",
                (outcome, finished, finished),
            )
        connection.commit()

    history = status(client)

    assert cs_cl(history)["backfilled_through"] == "2025-12-03"
    assert datetime.fromisoformat(cs_cl(history)["updated_through"]) == datetime(
        2025, 12, 3, 18, tzinfo=UTC
    )
    assert datetime.fromisoformat(history["last_daily_collection_at"]) == datetime(
        2025, 12, 5, 6, tzinfo=UTC
    )


def test_days_without_papers_inside_collected_history_are_listed_and_counted(client, collected):
    history = cs_cl(status(client))

    assert history["empty_day_count"] == 29
    assert len(history["empty_days"]) == 29
    assert history["empty_days"][:3] == ["2025-11-02", "2025-11-04", "2025-11-05"]
    assert history["empty_days"][-2:] == ["2025-11-30", "2025-12-02"]


def test_days_after_backfill_boundary_are_not_counted_as_empty(client, collected):
    assert "2025-12-04" not in cs_cl(status(client))["empty_days"]


def test_slice_without_collection_state_has_no_boundary_and_no_empty_days(client, database_url):
    history = cs_cl(status(client))

    assert history["backfilled_through"] is None
    assert history["updated_through"] is None
    assert history["months"] == []
    assert history["empty_days"] == []
    assert history["empty_day_count"] == 0


def test_month_inside_history_without_papers_is_shown_with_zero(client, database_url):
    with connect(database_url) as connection:
        set_state(connection, "cs.CL", "2026-01-31", None)
        add_paper(connection, "2511.00001", "2025-11-03", ["cs.CL"])
        add_paper(connection, "2601.00001", "2026-01-05", ["cs.CL"])
        connection.commit()

    months = cs_cl(status(client))["months"]

    assert [(m["month"], m["papers"]) for m in months] == [
        ("2025-11", 1),
        ("2025-12", 0),
        ("2026-01", 1),
    ]


def test_papers_after_the_boundary_are_counted_in_a_partial_month(client, database_url):
    with connect(database_url) as connection:
        set_state(connection, "cs.CL", "2025-11-30", None)
        add_paper(connection, "2512.00001", "2025-12-09", ["cs.CL"])
        connection.commit()

    months = cs_cl(status(client))["months"]

    assert [(m["month"], m["papers"], m["partial"]) for m in months] == [
        ("2025-11", 0, False),
        ("2025-12", 1, True),
    ]


def test_service_status_reports_database_size(client, database_url):
    history = status(client)

    assert history["database_size_bytes"] > 0
    assert history["last_daily_collection_at"] is None


def test_collection_by_the_daily_job_extends_the_collected_history(client, database_url):
    with connect(database_url) as connection:
        set_state(connection, "cs.CL", "2025-11-30", "2026-01-02T10:00Z")
        for number in range(1, 32):
            if number != 10:
                add_paper(connection, f"2512.{number:05d}", f"2025-12-{number:02d}", ["cs.CL"])
        connection.commit()

    found = cs_cl(status(client))

    assert found["backfilled_through"] == "2025-11-30"
    assert found["collected_through"] == "2026-01-01"
    assert "2025-12-10" in found["empty_days"]
    assert [d for d in found["empty_days"] if d.startswith("2025-12")] == ["2025-12-10"]
    partial = {m["month"]: m["partial"] for m in found["months"]}
    assert partial["2025-12"] is False
    assert partial["2026-01"] is True


def test_collected_through_is_the_backfill_boundary_when_it_is_later(client, collected):
    assert cs_cl(status(client))["collected_through"] == "2025-12-03"


def test_months_run_up_to_the_collected_through_month(client, database_url):
    with connect(database_url) as connection:
        set_state(connection, "cs.CL", "2026-01-31", None)
        add_paper(connection, "2511.00001", "2025-11-03", ["cs.CL"])
        connection.commit()

    months = cs_cl(status(client))["months"]

    assert [(m["month"], m["papers"]) for m in months] == [
        ("2025-11", 1),
        ("2025-12", 0),
        ("2026-01", 0),
    ]


def test_failed_history_summary_does_not_hide_the_latest_runs(
    client, database_url, fake_arxiv, monkeypatch
):
    collect.run(database_url, fake_arxiv)

    def broken(connection):
        raise psycopg.OperationalError("connection lost")

    monkeypatch.setattr(mcp_server, "history_summary", broken)

    response = call(client, "tools/call", {"name": "service_status", "arguments": {}})

    result = response.json()["result"]["structuredContent"]
    assert [run["job"] for run in result["latest_runs"]] == ["collect"]
    assert result["history"] is None
    assert result["history_error"] == "OperationalError: connection lost"


def test_query_failing_inside_the_snapshot_reports_its_own_error(client, database_url, monkeypatch):
    def failing_query(connection, category):
        connection.execute("SELECT 1 / 0")

    monkeypatch.setattr(history_summary, "_slice_history", failing_query)

    response = call(client, "tools/call", {"name": "service_status", "arguments": {}})

    result = response.json()["result"]["structuredContent"]
    assert result["history"] is None
    assert result["history_error"].startswith("DivisionByZero:")
