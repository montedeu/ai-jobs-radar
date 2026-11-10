"""Reads papers from the arXiv API (export.arxiv.org/api/query).

The API returns the latest version of each paper together with the date of its first
version (``published``), which is the Paper's date everywhere in this project.
"""

import re
import time
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Iterator
from datetime import date, datetime

import httpx
from pydantic import BaseModel

API_URL = "https://export.arxiv.org/api/query"
# arXiv's terms: at most one request every three seconds, over a single connection,
# shared by everything that runs this code.
SECONDS_BETWEEN_REQUESTS = 3.0
USER_AGENT = "ml-research-radar"
REQUEST_TIMEOUT_SECONDS = 60
# A page that fails, or comes back empty although more results exist (the API does this
# under load), is asked for again after a growing pause.
ATTEMPTS = 5
PAGE_SIZE = 200

NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}
_ID = re.compile(r"arxiv\.org/abs/(?P<id>.+?)v(?P<version>\d+)$")
_WHITESPACE = re.compile(r"\s+")


class Paper(BaseModel):
    arxiv_id: str  # without the version suffix
    version: int
    title: str
    abstract: str
    authors: list[str]
    primary_category: str
    categories: list[str]  # the primary one included
    first_version_date: date
    arxiv_updated_at: datetime


class Page(BaseModel):
    total: int
    papers: list[Paper]


class ArxivUnavailable(RuntimeError):
    pass


def _text(element: ElementTree.Element, path: str) -> str:
    # Titles and abstracts arrive hard-wrapped; the line breaks carry no meaning.
    return _WHITESPACE.sub(" ", element.findtext(path, "", NAMESPACES)).strip()


def parse_feed(xml: str) -> Page:
    feed = ElementTree.fromstring(xml)
    papers = []
    for entry in feed.findall("atom:entry", NAMESPACES):
        match = _ID.search(_text(entry, "atom:id"))
        if not match:
            raise ValueError(f"unexpected entry id: {_text(entry, 'atom:id')!r}")
        primary = entry.find("arxiv:primary_category", NAMESPACES).attrib["term"]
        listed = [c.attrib["term"] for c in entry.findall("atom:category", NAMESPACES)]
        papers.append(
            Paper(
                arxiv_id=match["id"],
                version=int(match["version"]),
                title=_text(entry, "atom:title"),
                abstract=_text(entry, "atom:summary"),
                authors=[_text(a, "atom:name") for a in entry.findall("atom:author", NAMESPACES)],
                primary_category=primary,
                categories=list(dict.fromkeys([primary, *listed])),
                first_version_date=datetime.fromisoformat(_text(entry, "atom:published")).date(),
                arxiv_updated_at=datetime.fromisoformat(_text(entry, "atom:updated")),
            )
        )
    return Page(total=int(_text(feed, "opensearch:totalResults")), papers=papers)


class ArxivClient:
    """One connection, one request every three seconds, retries with a growing pause."""

    def __init__(
        self,
        http: httpx.Client | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=1),
        )
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self.requests = 0

    def _get(self, params: dict[str, str | int]) -> str:
        if self._last_request_at is not None:
            wait = SECONDS_BETWEEN_REQUESTS - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()
        self.requests += 1
        response = self._http.get(API_URL, params=params)
        response.raise_for_status()
        return response.text

    def _page(self, query: str, sort_by: str, order: str, start: int) -> Page:
        params = {
            "search_query": query,
            "sortBy": sort_by,
            "sortOrder": order,
            "start": start,
            "max_results": PAGE_SIZE,
        }
        problem = ""
        for attempt in range(1, ATTEMPTS + 1):
            try:
                page = parse_feed(self._get(params))
            except (httpx.HTTPError, ElementTree.ParseError) as error:
                problem = f"{type(error).__name__}: {error}"
            else:
                if page.papers or start >= page.total:
                    return page
                problem = f"empty page at {start} of {page.total}"
            if attempt < ATTEMPTS:
                self._sleep(SECONDS_BETWEEN_REQUESTS * 2**attempt)
        raise ArxivUnavailable(f"{query!r} from {start}: {problem} ({ATTEMPTS} attempts)")

    def _pages(self, query: str, sort_by: str, order: str) -> Iterator[Paper]:
        start = 0
        while True:
            page = self._page(query, sort_by, order, start)
            yield from page.papers
            start += len(page.papers)
            if not page.papers or start >= page.total:
                return

    def submitted_on(self, category: str, day: date) -> Iterator[Paper]:
        """Papers of the slice whose first version was submitted on ``day`` (UTC)."""
        stamp = day.strftime("%Y%m%d")
        query = f"cat:{category} AND submittedDate:[{stamp}0000 TO {stamp}2359]"
        return self._pages(query, "submittedDate", "ascending")

    def latest_updated(self, category: str) -> Iterator[Paper]:
        """Papers of the slice, most recently updated first: new papers and new versions.

        The API's ``lastUpdatedDate`` range filter selects by submission date, so it
        cannot find revisions; sorting by it does, and the caller stops reading where
        its previous run began.
        """
        return self._pages(f"cat:{category}", "lastUpdatedDate", "descending")
