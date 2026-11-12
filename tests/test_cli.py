import pytest

from research_radar import cli, collect, daily_extraction
from research_radar.settings import EXTRACTION_DAILY_MAX_PAPERS


def test_max_papers_reaches_the_collect_job(monkeypatch):
    seen = {}
    monkeypatch.setattr(collect, "run", lambda **kwargs: seen.update(kwargs))

    cli.main(["collect", "--max-papers", "20000"])

    assert seen == {"max_papers": 20000}


def test_collect_without_the_flag_keeps_the_default_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(collect, "run", lambda **kwargs: seen.update(kwargs))

    cli.main(["collect"])

    assert seen == {"max_papers": None}


@pytest.mark.parametrize("value", ["0", "-5", "many"])
def test_max_papers_must_be_a_positive_number(value, capsys):
    with pytest.raises(SystemExit):
        cli.main(["collect", "--max-papers", value])
    assert "--max-papers" in capsys.readouterr().err


def test_check_jobs_runs_the_check(monkeypatch):
    called = []
    monkeypatch.setattr(collect, "check", lambda: called.append(True))

    cli.main(["check-jobs"])

    assert called == [True]


def test_daily_runs_collection_then_extraction_with_the_raised_limit(monkeypatch):
    order = []
    monkeypatch.setattr(collect, "run", lambda **kwargs: order.append(("collect", kwargs)))
    monkeypatch.setattr(
        daily_extraction, "run", lambda **kwargs: order.append(("extract", kwargs["max_papers"]))
    )

    cli.main(["daily", "--max-extract-papers", "900", "--max-papers", "20000"])

    assert order == [("collect", {"max_papers": 20000}), ("extract", 900)]


def test_daily_without_max_papers_keeps_the_collect_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(collect, "run", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(daily_extraction, "run", lambda **kwargs: None)

    cli.main(["daily"])

    assert seen == {"max_papers": None}


def test_daily_without_the_flag_keeps_the_default_extraction_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(collect, "run", lambda **kwargs: None)
    monkeypatch.setattr(daily_extraction, "run", lambda **kwargs: seen.update(kwargs))

    cli.main(["daily"])

    assert seen["max_papers"] == EXTRACTION_DAILY_MAX_PAPERS


@pytest.mark.parametrize("value", ["0", "-5", "many"])
def test_max_extract_papers_must_be_a_positive_number(value, capsys):
    with pytest.raises(SystemExit):
        cli.main(["daily", "--max-extract-papers", value])
    assert "--max-extract-papers" in capsys.readouterr().err
