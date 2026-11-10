from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from research_radar.arxiv_api import (
    ATTEMPTS,
    SECONDS_BETWEEN_REQUESTS,
    ArxivClient,
    ArxivUnavailable,
    parse_feed,
)

FEED = (Path(__file__).parent / "fixtures" / "arxiv_feed.xml").read_text()
EMPTY_FEED = FEED[: FEED.index("<entry>")] + "</feed>"


def feed_with_total(total: int, *, entries: bool = True) -> str:
    body = FEED if entries else EMPTY_FEED
    return body.replace("<opensearch:totalResults>124917<", f"<opensearch:totalResults>{total}<")


class Recorder:
    """An httpx transport that answers from a script and a clock that sleeping advances."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []
        self.now = 0.0
        self.slept = []

    def handler(self, request):
        self.requests.append((self.now, dict(request.url.params)))
        answer = self.answers.pop(0)
        if isinstance(answer, int):
            return httpx.Response(answer)
        return httpx.Response(200, text=answer)

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds

    def client(self):
        http = httpx.Client(transport=httpx.MockTransport(self.handler))
        return ArxivClient(http, sleep=self.sleep, clock=lambda: self.now)


@pytest.fixture(autouse=True)
def no_real_arxiv():
    """Overrides the suite-wide guard: these tests use a mock transport instead."""


def test_feed_is_parsed_into_papers():
    page = parse_feed(FEED)

    assert page.total == 124917
    assert [p.arxiv_id for p in page.papers] == [
        "2611.10486",
        "2611.10469",
        "2605.03780",
        "2510.11052",
    ]
    assert [p.version for p in page.papers] == [1, 1, 2, 3]


def test_paper_date_is_the_first_version_not_the_latest():
    revised = parse_feed(FEED).papers[2]

    assert revised.version == 2
    assert revised.first_version_date == date(2026, 5, 5)
    assert revised.arxiv_updated_at == datetime(2026, 11, 7, 5, 55, tzinfo=UTC)


def test_categories_start_with_the_primary_one_without_duplicates():
    revised = parse_feed(FEED).papers[2]

    assert revised.primary_category == "cs.LG"
    assert revised.categories[0] == "cs.LG"
    assert "cs.CL" in revised.categories
    assert len(revised.categories) == len(set(revised.categories))


def test_title_and_abstract_lose_their_hard_line_breaks():
    for paper in parse_feed(FEED).papers:
        assert "\n" not in paper.title + paper.abstract
        assert "  " not in paper.title + paper.abstract
        assert paper.abstract and paper.authors


def test_requests_are_at_least_three_seconds_apart():
    recorder = Recorder([feed_with_total(6), feed_with_total(6)])

    list(recorder.client().latest_updated("cs.CL"))

    first, second = (at for at, _ in recorder.requests)
    assert second - first >= SECONDS_BETWEEN_REQUESTS


def test_paging_continues_from_the_number_of_papers_received():
    recorder = Recorder([feed_with_total(6), feed_with_total(6)])

    papers = list(recorder.client().latest_updated("cs.CL"))

    assert len(papers) == 8
    assert [params["start"] for _, params in recorder.requests] == ["0", "4"]


def test_latest_updated_sorts_by_update_newest_first():
    recorder = Recorder([feed_with_total(4)])

    list(recorder.client().latest_updated("cs.CL"))

    ((_, params),) = recorder.requests
    assert (params["search_query"], params["sortBy"], params["sortOrder"]) == (
        "cat:cs.CL",
        "lastUpdatedDate",
        "descending",
    )


def test_submitted_on_asks_for_one_utc_day():
    recorder = Recorder([feed_with_total(4)])

    list(recorder.client().submitted_on("cs.CL", date(2026, 11, 5)))

    ((_, params),) = recorder.requests
    assert params["search_query"] == "cat:cs.CL AND submittedDate:[202611050000 TO 202611052359]"


def test_day_without_papers_is_not_an_error():
    recorder = Recorder([feed_with_total(0, entries=False)])

    assert list(recorder.client().submitted_on("cs.CL", date(2026, 11, 8))) == []
    assert len(recorder.requests) == 1


@pytest.mark.parametrize("failure", [503, 429, feed_with_total(50, entries=False), "<broken"])
def test_failed_or_wrongly_empty_page_is_asked_for_again(failure):
    recorder = Recorder([failure, feed_with_total(4)])

    assert len(list(recorder.client().latest_updated("cs.CL"))) == 4
    assert len(recorder.requests) == 2


def test_gives_up_after_the_allowed_attempts():
    recorder = Recorder([503] * ATTEMPTS)

    with pytest.raises(ArxivUnavailable, match="503"):
        list(recorder.client().latest_updated("cs.CL"))
    assert len(recorder.requests) == ATTEMPTS
