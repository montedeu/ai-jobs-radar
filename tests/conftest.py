import importlib.util
import os

import pytest

from research_radar import migrate


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """Tests never touch the real database, even with DATABASE_URL exported."""
    monkeypatch.delenv("DATABASE_URL", raising=False)


@pytest.fixture(scope="session")
def postgres_url(tmp_path_factory):
    """``$TEST_DATABASE_URL`` (CI's service container), else an embedded Postgres."""
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pgdata"), cleanup_mode="stop")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture
def database_url(postgres_url):
    """A database migrated to the latest schema and emptied of the previous test's rows."""
    migrate.downgrade(postgres_url, "base")
    migrate.upgrade(postgres_url)
    return postgres_url


class FakeArxiv:
    """Stands in for ArxivClient: serves the Papers it was given, and counts requests."""

    def __init__(self, papers=()):
        self.papers = list(papers)
        self.requests = 0
        self.days_asked = []

    def latest_updated(self, category):
        self.requests += 1
        matching = [p for p in self.papers if category in p.categories]
        return iter(sorted(matching, key=lambda p: p.arxiv_updated_at, reverse=True))

    def submitted_on(self, category, day):
        self.requests += 1
        self.days_asked.append(day)
        return iter(
            p for p in self.papers if category in p.categories and p.first_version_date == day
        )


@pytest.fixture
def fake_arxiv():
    return FakeArxiv()


@pytest.fixture(autouse=True)
def no_real_arxiv(monkeypatch):
    """A test that forgets to pass a fake client fails instead of calling arXiv."""

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach arXiv")

    monkeypatch.setattr("httpx.Client.get", refuse)


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    """A test that forgets to pass a fake model client fails instead of calling the API.

    The client is refused where it is built, so neither our wrapper nor the SDK's own
    client can open a connection. The refusal sits on the SDK's shared base client as well
    as on Anthropic, so the Bedrock, Vertex and Foundry clients are refused too. The SDK
    import is optional: the guard must not break collection where ``anthropic`` is not
    installed.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("a test tried to reach the Anthropic API")

    monkeypatch.setattr("research_radar.extraction.AnthropicModelClient.__init__", refuse)
    if importlib.util.find_spec("anthropic"):
        monkeypatch.setattr("anthropic.Anthropic.__init__", refuse)
        monkeypatch.setattr("anthropic.AsyncAnthropic.__init__", refuse)
        monkeypatch.setattr("anthropic._base_client.SyncAPIClient.__init__", refuse)
        monkeypatch.setattr("anthropic._base_client.AsyncAPIClient.__init__", refuse)
