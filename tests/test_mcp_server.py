import pytest
from starlette.testclient import TestClient

from research_radar import collect, mcp_server

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


@pytest.mark.parametrize("token", [None, "wrong", TOKEN + "x", ""])
def test_mcp_endpoint_rejects_a_missing_or_wrong_token(client, token):
    assert call(client, "tools/list", token=token).status_code == 401


def test_unknown_paths_are_rejected_before_routing(client):
    assert client.get("/anything").status_code == 401


def test_service_status_reports_the_latest_collect_run(client, database_url):
    collect.run(database_url)
    response = call(client, "tools/call", {"name": "service_status", "arguments": {}})
    assert response.status_code == 200
    (run,) = response.json()["result"]["structuredContent"]["latest_runs"]
    assert (run["job"], run["status"]) == ("collect", "succeeded")
