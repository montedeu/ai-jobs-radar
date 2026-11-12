"""The synonym dictionary is loaded at deploy and each change of its content is a version."""

import hashlib
import json
import threading
from importlib import resources

import psycopg
import pytest

from research_radar import cli, database
from research_radar import synonyms as synonyms_module
from research_radar.synonyms import (
    SYNONYMS_FILE,
    current_dictionary,
    load_synonyms,
    read_synonyms,
)

HEADER = "normal_form,canonical_name\n"
ONE_ENTRY = HEADER + "lora,LoRA\n"
TWO_ENTRIES = ONE_ENTRY + "low rank adaptation,LoRA\n"


def write(tmp_path, text):
    path = tmp_path / "synonyms.csv"
    path.write_text(text, encoding="utf-8")
    return path


def content_hash(path) -> str:
    pairs = sorted(read_synonyms(path).items())
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def version_count(connection) -> int:
    return connection.execute("SELECT count(*) FROM synonym_version").fetchone()[0]


def entries(connection) -> set[str]:
    return {row[0] for row in connection.execute("SELECT normal_form FROM synonym")}


def test_first_load_records_a_version(database_url, tmp_path):
    path = write(tmp_path, ONE_ENTRY)

    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, path)
        version = current_dictionary(connection)

    assert version is not None
    assert version.file_sha256 == content_hash(path)
    assert version.entries == 1
    assert version.loaded_at is not None


def test_loading_the_same_file_again_adds_no_version(database_url, tmp_path):
    path = write(tmp_path, ONE_ENTRY)

    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, path)
        first = current_dictionary(connection)
        load_synonyms(connection, path)
        second = current_dictionary(connection)

        assert version_count(connection) == 1
    assert second == first  # loaded_at is the date of the change, not of the deploy


def test_changed_file_records_a_new_version(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, ONE_ENTRY))
        first = current_dictionary(connection)
        load_synonyms(connection, write(tmp_path, TWO_ENTRIES))
        second = current_dictionary(connection)

        assert version_count(connection) == 2
        assert entries(connection) == {"lora", "low rank adaptation"}
    assert second.id != first.id
    assert second.file_sha256 != first.file_sha256
    assert second.entries == 2
    assert second.loaded_at >= first.loaded_at


def test_returning_to_an_earlier_file_records_a_version(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, ONE_ENTRY))
        load_synonyms(connection, write(tmp_path, TWO_ENTRIES))
        load_synonyms(connection, write(tmp_path, ONE_ENTRY))

        assert version_count(connection) == 3
        assert current_dictionary(connection).entries == 1


def test_invalid_file_changes_neither_table_nor_versions(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, ONE_ENTRY))
        before = current_dictionary(connection)

        broken = write(tmp_path, HEADER + "lora,LoRA\nLoRA,x\n")
        with pytest.raises(ValueError):
            load_synonyms(connection, broken)

        assert entries(connection) == {"lora"}
        assert current_dictionary(connection) == before
        assert version_count(connection) == 1


def test_a_failure_after_the_version_row_rolls_the_version_back(
    database_url, tmp_path, monkeypatch
):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, ONE_ENTRY))
        changed = write(tmp_path, TWO_ENTRIES)

        # Postgres refuses a NUL in text, so the insert into synonym fails after the
        # version row is written: the version must go back with the table.
        monkeypatch.setattr(synonyms_module, "read_synonyms", lambda path: {"a\x00": "x"})
        with pytest.raises(psycopg.Error):
            load_synonyms(connection, changed)

        assert entries(connection) == {"lora"}
        assert version_count(connection) == 1


def test_current_dictionary_is_none_before_any_load(database_url):
    with psycopg.connect(database_url) as connection:
        assert current_dictionary(connection) is None


def test_header_only_file_is_a_version_with_no_entries(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, TWO_ENTRIES))
        assert load_synonyms(connection, write(tmp_path, HEADER)) == 0

        assert entries(connection) == set()
        assert current_dictionary(connection).entries == 0
        assert version_count(connection) == 2


def test_two_loads_at_once_record_one_version(database_url, tmp_path):
    path = write(tmp_path, ONE_ENTRY)
    results = []
    with psycopg.connect(database_url) as first, psycopg.connect(database_url) as second:
        worker = threading.Thread(target=lambda: results.append(load_synonyms(second, path)))
        with first.transaction():
            load_synonyms(first, path)
            worker.start()
            worker.join(timeout=0.5)
            assert worker.is_alive()  # the second load waits for the first one's transaction
        worker.join(timeout=10)
        assert not worker.is_alive()

        assert results == [1]
        assert version_count(first) == 1


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_cli_load_synonyms_loads_the_packaged_file(database_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", database_url)

    cli.main(["load-synonyms"])

    expected = read_synonyms(SYNONYMS_FILE)
    with psycopg.connect(database_url) as connection:
        assert entries(connection) == set(expected)
        version = current_dictionary(connection)
    assert version.entries == len(expected)
    assert version.file_sha256 == content_hash(SYNONYMS_FILE)


def test_cli_load_synonyms_connects_to_the_direct_address(monkeypatch):
    pooled = "postgresql://u:p@ep-cool-123-pooler.example.neon.tech/db"
    monkeypatch.setenv("DATABASE_URL", pooled)
    seen = []

    monkeypatch.setattr(database, "connect", lambda url=None: seen.append(url) or FakeConnection())
    monkeypatch.setattr(synonyms_module, "load_synonyms", lambda connection: 0)

    cli.main(["load-synonyms"])

    assert seen == [database.direct_url(pooled)]
    assert "-pooler" not in seen[0]


def test_packaged_synonyms_file_is_in_the_installed_package():
    # The image installs the built package and has no source tree; the file must be
    # resolved through the package, not through the repository layout.
    resource = resources.files("research_radar") / "data" / "synonyms.csv"

    assert resource.is_file()
    with resources.as_file(resource) as path:
        assert isinstance(read_synonyms(path), dict)


def test_reordered_lines_and_other_line_endings_add_no_version(database_url, tmp_path):
    with psycopg.connect(database_url) as connection:
        load_synonyms(connection, write(tmp_path, TWO_ENTRIES))
        first = current_dictionary(connection)

        reordered = HEADER + "low rank adaptation,LoRA\nlora,LoRA\n"
        load_synonyms(connection, write(tmp_path, reordered))
        path = tmp_path / "crlf.csv"
        path.write_bytes(TWO_ENTRIES.replace("\n", "\r\n").encode())
        load_synonyms(connection, path)

        assert version_count(connection) == 1
        assert current_dictionary(connection) == first


def test_cli_load_synonyms_reports_an_invalid_file_in_one_line(monkeypatch, tmp_path, capsys):
    broken = write(tmp_path, HEADER + "lora,LoRA\nlora,x\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
    monkeypatch.setattr(synonyms_module, "SYNONYMS_FILE", broken)
    monkeypatch.setattr(
        synonyms_module, "load_synonyms", lambda c: synonyms_module.read_synonyms(broken)
    )
    monkeypatch.setattr(database, "connect", lambda url=None: FakeConnection())

    with pytest.raises(SystemExit) as raised:
        cli.main(["load-synonyms"])

    assert str(raised.value).startswith("error: ")
    assert "\n" not in str(raised.value)
    assert "lora" in str(raised.value)
