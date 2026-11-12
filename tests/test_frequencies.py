"""The table of the most frequent entities of an extraction run.

Fixture: run 1 has four cs.CL papers with a completion mark and one without; LoRA (method)
is in three of the four, BERT (model) in two, machine translation (task) in one. Run 2 has
LoRA in four papers and a cs.CV paper has BERT: neither may change run 1's numbers.
"""

import psycopg
import pytest

from research_radar import cli
from research_radar.frequencies import top_entities, write_table

LORA = ("method", "LoRA")
BERT = ("model", "BERT")
MT = ("task", "machine translation")


class Corpus:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection
        self.run_id = self.new_run()
        self._papers = 0

    def new_run(self) -> int:
        return self.connection.execute(
            "INSERT INTO extraction_run (model, prompt_hash) VALUES ('test', 'test') RETURNING id"
        ).fetchone()[0]

    def paper(
        self,
        entities=(),
        *,
        category="cs.CL",
        extracted=True,
        run_id=None,
        title=None,
        abstract="a",
    ):
        """One paper; returns its arXiv id."""
        self._papers += 1
        arxiv_id = f"p{self._papers:04d}"
        self.connection.execute(
            """
            INSERT INTO paper (arxiv_id, version, title, abstract, abstract_hash, authors,
                               primary_category, first_version_date, arxiv_updated_at)
            VALUES (%s, 1, %s, %s, 'h', '[]', %s, '2026-10-01', '2026-10-01')
            """,
            (arxiv_id, title or f"title {arxiv_id}", abstract, category),
        )
        self.connection.execute("INSERT INTO paper_category VALUES (%s, %s)", (arxiv_id, category))
        if extracted:
            run_id = run_id or self.run_id
            self.connection.execute(
                "INSERT INTO paper_extraction (run_id, arxiv_id, abstract_hash, input_tokens,"
                " output_tokens) VALUES (%s, %s, 'h', 0, 0)",
                (run_id, arxiv_id),
            )
            for entity_type, name in entities:
                self.mention(arxiv_id, entity_type, name, run_id=run_id)
        return arxiv_id

    def mention(self, arxiv_id, entity_type, spelling, *, run_id=None, normal_form=None):
        self.connection.execute(
            "INSERT INTO mention (run_id, arxiv_id, entity_type, spelling, normal_form, expansion)"
            " VALUES (%s, %s, %s, %s, %s, NULL)",
            (
                run_id or self.run_id,
                arxiv_id,
                entity_type,
                spelling,
                normal_form or spelling.lower(),
            ),
        )

    def top(self, **kwargs):
        return top_entities(self.connection, kwargs.pop("run_id", self.run_id), **kwargs)


@pytest.fixture
def corpus(database_url):
    with psycopg.connect(database_url) as connection:
        yield Corpus(connection)


@pytest.fixture
def fixture_corpus(corpus):
    corpus.paper([LORA, BERT, MT])
    corpus.paper([LORA, BERT])
    corpus.paper([LORA])
    corpus.paper([])
    corpus.paper([LORA, BERT, MT], extracted=False)
    other_run = corpus.new_run()
    for _ in range(4):
        corpus.paper([LORA], run_id=other_run)
    corpus.paper([BERT], category="cs.CV")
    return corpus


def names(rows):
    return [row.canonical_name for row in rows]


def test_top_entities_are_ordered_by_share_of_papers(fixture_corpus):
    rows = fixture_corpus.top()

    assert [(r.canonical_name, r.entity_type, r.papers, r.share) for r in rows] == [
        ("lora", "method", 3, 0.75),
        ("bert", "model", 2, 0.5),
        ("machine translation", "task", 1, 0.25),
    ]


def test_share_denominator_is_papers_with_completion_mark_in_the_run(corpus):
    corpus.paper([LORA])
    corpus.paper([])
    corpus.paper([LORA], extracted=False)

    (row,) = corpus.top()

    assert (row.papers, row.share) == (1, 0.5)


def test_paper_with_two_spellings_counts_once(corpus):
    corpus.connection.execute(
        "INSERT INTO synonym (normal_form, canonical_name) VALUES"
        " ('rag', 'rag'), ('retrieval-augmented generation', 'rag')"
    )
    paper = corpus.paper([])
    corpus.mention(paper, "method", "RAG")
    corpus.mention(paper, "method", "retrieval-augmented generation")
    corpus.paper([])

    (row,) = corpus.top()

    assert (row.canonical_name, row.papers, row.share) == ("rag", 1, 0.5)
    assert sorted(row.spellings) == ["RAG", "retrieval-augmented generation"]


def test_spellings_are_ordered_by_frequency(corpus):
    for spelling in ("LoRA", "LoRA", "Lora"):
        paper = corpus.paper([])
        corpus.mention(paper, "method", spelling, normal_form="lora")

    (row,) = corpus.top()

    assert row.spellings == ["LoRA", "Lora"]


def test_other_run_and_other_slice_are_not_counted(fixture_corpus):
    rows = fixture_corpus.top()
    bert = next(r for r in rows if r.canonical_name == "bert")

    assert bert.papers == 2
    assert next(r for r in rows if r.canonical_name == "lora").papers == 3
    assert [(r.canonical_name, r.share) for r in fixture_corpus.top(category="cs.CV")] == [
        ("bert", 1.0)
    ]


def test_limit_keeps_the_top_rows(fixture_corpus):
    assert names(fixture_corpus.top(limit=2)) == ["lora", "bert"]


def test_ties_are_ordered_by_canonical_name(corpus):
    corpus.paper([("method", "zeta"), ("method", "alpha")])

    assert names(corpus.top()) == ["alpha", "zeta"]


def test_each_row_has_up_to_n_examples(fixture_corpus):
    rows = {r.canonical_name: r for r in fixture_corpus.top(examples=2)}

    assert [e.arxiv_id for e in rows["lora"].examples] == ["p0001", "p0002"]
    assert [e.arxiv_id for e in rows["machine translation"].examples] == ["p0001"]
    assert rows["bert"].examples[0].title == "title p0001"


def test_run_without_completion_marks_gives_no_rows(corpus):
    assert corpus.top(run_id=corpus.new_run()) == []
    assert corpus.top(run_id=9999) == []


def test_written_table_has_empty_recommendation_columns_and_examples(corpus, tmp_path):
    corpus.paper(
        [LORA], title="A | tricky title", abstract="first line\nsecond | line", category="cs.CL"
    )
    corpus.paper([])
    path = tmp_path / "frequencies.md"

    write_table(corpus.top(), path, run_id=corpus.run_id, category="cs.CL", completed=2)

    text = path.read_text()
    lines = text.splitlines()
    assert (
        "| # | имя | тип | статьи | доля | написания | фоновое | узкие соседи | рекомендация |"
        in lines
    )
    row = next(line for line in lines if line.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[:5] == ["1", "lora", "method", "1", "0.50"]
    assert cells[6:] == ["", "", ""]
    assert "tricky" not in row
    assert "p0001" in text and "A \\| tricky title" in text and "second | line" in text
    assert "прогон " + str(corpus.run_id) in lines[0]
    assert "Статей с отметкой завершения в прогоне: 2." in lines


def test_written_table_for_empty_run_says_so_in_the_header(corpus, tmp_path):
    path = tmp_path / "frequencies.md"

    write_table([], path, run_id=9999, category="cs.CL")

    assert "нет статей с отметкой завершения" in path.read_text()


def test_cli_frequencies_writes_the_file(fixture_corpus, database_url, monkeypatch, tmp_path):
    fixture_corpus.connection.commit()
    monkeypatch.setenv("DATABASE_URL", database_url)
    path = tmp_path / "out.md"

    cli.main(["frequencies", "--run", str(fixture_corpus.run_id), "--out", str(path)])

    text = path.read_text()
    assert "| 1 | lora | method | 3 | 0.75 |" in text
    assert "machine translation" in text
