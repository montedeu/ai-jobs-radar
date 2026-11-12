"""The synonym dictionary: a file in the repository, loaded into the ``synonym`` table.

The file in the repository is the source of truth; approval is the developer's commit
(ADR 0002), which the loader does not check. The ``paper_entity`` view joins the table, so
loading recomputes canonical names without an extraction run.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg

from research_radar.normalize import normal_form

SYNONYMS_FILE = Path(__file__).parent / "data" / "synonyms.csv"
_COLUMNS = ["normal_form", "canonical_name"]


@dataclass(frozen=True)
class DictionaryVersion:
    id: int
    loaded_at: datetime
    file_sha256: str
    entries: int


def read_synonyms(path: Path = SYNONYMS_FILE) -> dict[str, str]:
    """normal form -> canonical name. Raises ValueError on a file that must not be loaded."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != _COLUMNS:
            raise ValueError(f"{path.name}: header must be {','.join(_COLUMNS)}")
        synonyms: dict[str, str] = {}
        for row in reader:
            key, name = row["normal_form"], row["canonical_name"] or ""
            if not key or key != normal_form(key):
                raise ValueError(f"{path.name}: {key!r} is not a normal form")
            if key in synonyms:
                raise ValueError(f"{path.name}: {key!r} appears twice")
            if not name.strip():
                raise ValueError(f"{path.name}: {key!r} has an empty canonical name")
            synonyms[key] = name
    return synonyms


def load_synonyms(connection: psycopg.Connection, path: Path = SYNONYMS_FILE) -> int:
    """Replace the table's content with the file's and, if the file differs from the last
    loaded one, record a version; all or nothing. Returns the entry count."""
    synonyms = read_synonyms(path)
    # Of the parsed content, not of the bytes: reordered lines or other line endings are
    # not a change of the dictionary.
    pairs = sorted(synonyms.items())
    file_sha256 = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    with connection.transaction():
        # Two loads at once (deploys in a row) take turns, so the second sees the first's
        # version and adds none. The lock is released at the end of the transaction.
        connection.execute("SELECT pg_advisory_xact_lock(hashtext('synonym_load'))")
        last = current_dictionary(connection)
        connection.execute("DELETE FROM synonym")
        with connection.cursor() as cursor:
            cursor.executemany("INSERT INTO synonym VALUES (%s, %s)", list(synonyms.items()))
        if last is None or last.file_sha256 != file_sha256:
            connection.execute(
                "INSERT INTO synonym_version (file_sha256, entries) VALUES (%s, %s)",
                (file_sha256, len(synonyms)),
            )
    return len(synonyms)


def current_dictionary(connection: psycopg.Connection) -> DictionaryVersion | None:
    """The last recorded version of the dictionary, or None before the first load."""
    row = connection.execute(
        "SELECT id, loaded_at, file_sha256, entries FROM synonym_version ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return DictionaryVersion(*row) if row else None
