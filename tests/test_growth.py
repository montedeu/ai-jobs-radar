"""What sql/growth.sql must return. The conditions are in docs/growth-query.md.

Every test builds its own handful of papers. Windows are 10 days and the thresholds are
small so the data stays readable: with AS_OF = 31 October the last window is
22-31 October and the previous one is 12-21 October.
"""

from datetime import date, timedelta

import psycopg
import pytest

from research_radar.growth import GROWTH_SQL, entity_growth

pytestmark = pytest.mark.skipif(
    not GROWTH_SQL.exists(), reason="sql/growth.sql is the developer's to write"
)

AS_OF = date(2026, 10, 31)
WINDOW_DAYS = 10
LAST = AS_OF  # a day inside the last window
PREVIOUS = AS_OF - timedelta(days=WINDOW_DAYS)  # a day inside the previous window
LORA = ("method", "lora")


class Corpus:
    """Inserts papers, their slices and their entities for one extraction run."""

    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self.run_id = self.new_run()
        self._papers = 0

    def new_run(self) -> int:
        return self.connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash) VALUES ('test', 'test') RETURNING id"
        ).fetchone()[0]

    def paper(self, day, entities=(), *, categories=("cs.CL",), extracted=True, run_id=None):
        """One paper first submitted on ``day``. Returns its arXiv id."""
        self._papers += 1
        arxiv_id = f"p{self._papers:04d}"
        self.connection.execute(
            """
            INSERT INTO paper (arxiv_id, title, abstract, abstract_hash, authors,
                               primary_category, first_version_date)
            VALUES (%s, 't', 'a', 'h', '[]', %s, %s)
            """,
            (arxiv_id, categories[0], day),
        )
        for category in categories:
            self.connection.execute(
                "INSERT INTO paper_category VALUES (%s, %s)", (arxiv_id, category)
            )
        if extracted:
            self.extract(arxiv_id, entities, run_id=run_id)
        return arxiv_id

    def extract(self, arxiv_id, entities, *, run_id=None):
        run_id = run_id or self.run_id
        self.connection.execute(
            "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
            " output_tokens) VALUES (%s, %s, 'h', 0, 0)",
            (run_id, arxiv_id),
        )
        for entity_type, name in entities:
            self.connection.execute(
                "INSERT INTO mention VALUES (%s, %s, %s, %s, %s)",
                (run_id, arxiv_id, entity_type, name, name),
            )

    def papers(self, day, count, entities=(), **kwargs):
        for _ in range(count):
            self.paper(day, entities, **kwargs)

    def growth(self, *, category="cs.CL", min_mentions=3, new_max_previous=1, run_id=None):
        return entity_growth(
            self.connection,
            run_id=run_id or self.run_id,
            category=category,
            as_of=AS_OF,
            window_days=WINDOW_DAYS,
            min_mentions=min_mentions,
            new_max_previous=new_max_previous,
        )


@pytest.fixture
def corpus(database_url):
    with psycopg.connect(database_url) as connection:
        yield Corpus(connection)


def only(rows):
    assert len(rows) == 1, rows
    return rows[0]


def kinds(rows):
    return {(row.canonical_name, row.kind) for row in rows}


# --- the three kinds -----------------------------------------------------------------


def test_rising_entity_with_its_counts_shares_and_ratio(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 4, [LORA])
    corpus.papers(LAST, 6)

    row = only(corpus.growth())

    assert (row.canonical_name, row.entity_type, row.kind) == ("lora", "method", "rising")
    assert (row.mentions_last, row.mentions_previous) == (4, 2)
    assert (row.papers_last, row.papers_previous) == (10, 10)
    assert row.share_last == pytest.approx(0.4)
    assert row.share_previous == pytest.approx(0.2)
    assert row.ratio == pytest.approx(2.0)


def test_fading_entity(corpus):
    corpus.papers(PREVIOUS, 4, [LORA])
    corpus.papers(PREVIOUS, 6)
    corpus.papers(LAST, 1, [LORA])
    corpus.papers(LAST, 9)

    row = only(corpus.growth())

    assert row.kind == "fading"
    assert (row.mentions_last, row.mentions_previous) == (1, 4)
    assert row.ratio == pytest.approx(0.25)


def test_fading_entity_that_vanished_has_zero_last_window(corpus):
    corpus.papers(PREVIOUS, 4, [LORA])
    corpus.papers(LAST, 5)

    row = only(corpus.growth())

    assert (row.kind, row.mentions_last, row.share_last) == ("fading", 0, 0)
    assert row.ratio == pytest.approx(0.0)


def test_new_entity_absent_before_has_no_ratio(corpus):
    corpus.papers(PREVIOUS, 5)
    corpus.papers(LAST, 3, [LORA])
    corpus.papers(LAST, 2)

    row = only(corpus.growth())

    assert (row.kind, row.mentions_last, row.mentions_previous) == ("new", 3, 0)
    assert row.share_previous == 0
    assert row.ratio is None


def test_entity_below_the_new_threshold_before_is_new_not_rising(corpus):
    corpus.papers(PREVIOUS, 1, [LORA])
    corpus.papers(PREVIOUS, 9)
    corpus.papers(LAST, 5, [LORA])
    corpus.papers(LAST, 5)

    row = only(corpus.growth(new_max_previous=2))

    assert row.kind == "new"
    assert row.ratio == pytest.approx(5.0)


def test_entity_at_the_new_threshold_before_is_rising(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 5, [LORA])
    corpus.papers(LAST, 5)

    assert only(corpus.growth(new_max_previous=2)).kind == "rising"


# --- thresholds apply to the entity's own window ---------------------------------------


def test_growth_below_the_minimum_in_the_last_window_is_not_reported(corpus):
    corpus.papers(PREVIOUS, 1, [LORA])
    corpus.papers(PREVIOUS, 9)
    corpus.papers(LAST, 2, [LORA])
    corpus.papers(LAST, 8)

    assert corpus.growth(min_mentions=3) == []


def test_many_mentions_before_do_not_make_a_shrinking_entity_rising(corpus):
    # 19 and 1 pass a threshold on the sum of both windows; that is fading, never rising.
    corpus.papers(PREVIOUS, 19, [LORA])
    corpus.papers(LAST, 1, [LORA])
    corpus.papers(LAST, 18)

    assert kinds(corpus.growth(min_mentions=20)) == set()
    assert kinds(corpus.growth(min_mentions=19)) == {("lora", "fading")}


def test_decline_below_the_minimum_in_the_previous_window_is_not_reported(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 1, [LORA])
    corpus.papers(LAST, 9)

    assert corpus.growth(min_mentions=3) == []


def test_minimum_is_inclusive(corpus):
    corpus.papers(PREVIOUS, 1, [LORA])
    corpus.papers(PREVIOUS, 9)
    corpus.papers(LAST, 3, [LORA])
    corpus.papers(LAST, 7)

    assert only(corpus.growth(min_mentions=3)).kind == "rising"


def test_unchanged_share_is_neither_rising_nor_fading(corpus):
    for day in (PREVIOUS, LAST):
        corpus.papers(day, 3, [LORA])
        corpus.papers(day, 7)

    assert corpus.growth() == []


# --- a share is of the window's own papers ---------------------------------------------


def test_same_count_can_fade_when_the_last_window_has_more_papers(corpus):
    corpus.papers(PREVIOUS, 3, [LORA])
    corpus.papers(PREVIOUS, 2)
    corpus.papers(LAST, 3, [LORA])
    corpus.papers(LAST, 7)

    row = only(corpus.growth())

    assert row.kind == "fading"
    assert (row.papers_last, row.papers_previous) == (10, 5)
    assert row.share_last == pytest.approx(0.3)
    assert row.share_previous == pytest.approx(0.6)


def test_papers_not_yet_extracted_are_not_in_the_denominator(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 4, [LORA])
    corpus.papers(LAST, 6)
    corpus.papers(LAST, 30, extracted=False)  # collected today, batch still out

    row = only(corpus.growth())

    assert row.papers_last == 10
    assert row.share_last == pytest.approx(0.4)


def test_papers_extracted_only_by_another_run_are_not_counted(corpus):
    other_run = corpus.new_run()
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 4, [LORA])
    corpus.papers(LAST, 6)
    corpus.papers(LAST, 30, [LORA], run_id=other_run)

    row = only(corpus.growth())

    assert (row.mentions_last, row.papers_last) == (4, 10)


def test_same_paper_extracted_by_two_runs_counts_once_in_each(corpus):
    other_run = corpus.new_run()
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    for _ in range(4):
        corpus.extract(corpus.paper(LAST, [LORA]), [LORA], run_id=other_run)
    corpus.papers(LAST, 6)
    corpus.papers(PREVIOUS, 5, run_id=other_run)

    row = only(corpus.growth())
    other = only(corpus.growth(run_id=other_run))

    assert (row.mentions_last, row.papers_last) == (4, 10)
    assert (other.mentions_last, other.papers_last, other.papers_previous) == (4, 4, 5)


# --- slices ------------------------------------------------------------------------------


def test_papers_outside_the_slice_are_ignored(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 4, [LORA])
    corpus.papers(LAST, 6)
    corpus.papers(LAST, 50, [LORA], categories=("cs.CV",))

    row = only(corpus.growth(category="cs.CL"))

    assert (row.mentions_last, row.papers_last) == (4, 10)


def test_slice_includes_papers_where_the_category_is_secondary(corpus):
    corpus.papers(PREVIOUS, 2, [LORA])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 4, [LORA], categories=("cs.CV", "cs.CL"))
    corpus.papers(LAST, 6, categories=("stat.ML", "cs.LG", "cs.CL"))

    row = only(corpus.growth(category="cs.CL"))

    assert (row.mentions_last, row.papers_last) == (4, 10)


def test_paper_with_several_categories_is_one_paper(corpus):
    corpus.papers(PREVIOUS, 2, [LORA], categories=("cs.CL", "cs.LG", "cs.AI"))
    corpus.papers(PREVIOUS, 8, categories=("cs.CL", "cs.LG"))
    corpus.papers(LAST, 4, [LORA], categories=("cs.CL", "cs.LG", "cs.AI"))
    corpus.papers(LAST, 6, categories=("cs.CL", "cs.LG"))

    row = only(corpus.growth(category="cs.LG"))

    assert (row.mentions_last, row.papers_last) == (4, 10)
    assert (row.mentions_previous, row.papers_previous) == (2, 10)


# --- entities ------------------------------------------------------------------------------


def test_spellings_joined_by_the_dictionary_are_one_entity(corpus):
    corpus.connection.execute(
        "INSERT INTO synonym VALUES ('rag', 'retrieval-augmented generation')"
    )
    rag, long_form = ("method", "rag"), ("method", "retrieval-augmented generation")
    corpus.papers(PREVIOUS, 1, [rag])
    corpus.papers(PREVIOUS, 1, [long_form])
    corpus.papers(PREVIOUS, 8)
    corpus.papers(LAST, 2, [rag])
    corpus.papers(LAST, 1, [long_form])
    corpus.papers(LAST, 1, [rag, long_form])  # both spellings in one abstract: one mention
    corpus.papers(LAST, 6)

    row = only(corpus.growth())

    assert row.canonical_name == "retrieval-augmented generation"
    assert (row.mentions_last, row.mentions_previous) == (4, 2)


def test_same_name_with_different_types_is_two_entities(corpus):
    as_method, as_model = ("method", "bert"), ("model", "bert")
    corpus.papers(PREVIOUS, 4, [as_method])
    corpus.papers(PREVIOUS, 6)
    corpus.papers(LAST, 1, [as_method])
    corpus.papers(LAST, 4, [as_model])
    corpus.papers(LAST, 5)

    rows = corpus.growth()

    assert {(row.entity_type, row.kind) for row in rows} == {
        ("method", "fading"),
        ("model", "new"),
    }


def test_each_entity_is_judged_on_its_own(corpus):
    lora, rag, mmlu = LORA, ("method", "rag"), ("dataset", "mmlu")
    corpus.papers(PREVIOUS, 2, [lora, rag])
    corpus.papers(PREVIOUS, 2, [rag])
    corpus.papers(PREVIOUS, 6)
    corpus.papers(LAST, 4, [lora, mmlu])
    corpus.papers(LAST, 1, [rag])
    corpus.papers(LAST, 5)

    assert kinds(corpus.growth()) == {("lora", "rising"), ("rag", "fading"), ("mmlu", "new")}


# --- window boundaries ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("day", "window"),
    [
        (date(2026, 11, 1), None),  # the day after AS_OF: not yet
        (date(2026, 10, 31), "last"),  # AS_OF itself
        (date(2026, 10, 22), "last"),  # the first day of the last window
        (date(2026, 10, 21), "previous"),  # the last day of the previous window
        (date(2026, 10, 12), "previous"),  # the first day of the previous window
        (date(2026, 10, 11), None),  # one day too old
    ],
)
def test_window_boundaries(corpus, day, window):
    """A fixed backdrop makes LoRA fade; three more papers on ``day`` shift the counts."""
    corpus.papers(date(2026, 10, 15), 5, [LORA])
    corpus.papers(date(2026, 10, 25), 1, [LORA])
    corpus.papers(date(2026, 10, 25), 9)
    corpus.papers(day, 3, [LORA])

    row = only(corpus.growth())

    expected = {"last": (4, 5, 13, 5), "previous": (1, 8, 10, 8), None: (1, 5, 10, 5)}[window]
    assert (
        row.mentions_last,
        row.mentions_previous,
        row.papers_last,
        row.papers_previous,
    ) == expected


# --- order -----------------------------------------------------------------------------------


def test_order_is_rising_then_new_then_fading_strongest_first(corpus):
    def entity(name):
        return ("method", name)

    for name, previous, last in [
        ("rising-x2", 2, 4),
        ("rising-x3", 1, 3),
        ("rising-x3-b", 1, 3),  # tie on ratio and mentions: name decides
        ("new-big", 0, 5),
        ("new-small", 0, 3),
        ("fading-mild", 4, 3),
        ("fading-hard", 4, 1),
    ]:
        corpus.papers(PREVIOUS, previous, [entity(name)])
        corpus.papers(LAST, last, [entity(name)])
    corpus.papers(PREVIOUS, 8)  # 20 papers before, 22 after

    rows = corpus.growth()

    assert [row.canonical_name for row in rows] == [
        "rising-x3",
        "rising-x3-b",
        "rising-x2",
        "new-big",
        "new-small",
        "fading-hard",
        "fading-mild",
    ]
