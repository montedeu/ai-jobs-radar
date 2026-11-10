import pytest

from research_radar import cli, collect


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
