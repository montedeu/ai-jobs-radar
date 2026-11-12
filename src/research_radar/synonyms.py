"""The synonym dictionary: a file in the repository, loaded into the ``synonym`` table.

The file is the source of truth: an entry takes effect once the developer has committed it
(ADR 0002). The ``paper_entity`` view joins the table, so loading recomputes canonical
names without an extraction run.
"""

import csv
from pathlib import Path

import psycopg

from research_radar.normalize import normal_form

SYNONYMS_FILE = Path(__file__).parent / "data" / "synonyms.csv"
_COLUMNS = ["normal_form", "canonical_name"]


def read_synonyms(path: Path = SYNONYMS_FILE) -> dict[str, str]:
    """normal form -> canonical name. Raises ValueError on a file that must not be loaded."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != _COLUMNS:
            raise ValueError(f"{path.name}: header must be {','.join(_COLUMNS)}")
        synonyms: dict[str, str] = {}
        for row in reader:
            key, name = row["normal_form"], (row["canonical_name"] or "").strip()
            if not key or key != normal_form(key):
                raise ValueError(f"{path.name}: {key!r} is not a normal form")
            if key in synonyms:
                raise ValueError(f"{path.name}: {key!r} appears twice")
            if not name:
                raise ValueError(f"{path.name}: {key!r} has an empty canonical name")
            synonyms[key] = name
    return synonyms


def load_synonyms(connection: psycopg.Connection, path: Path = SYNONYMS_FILE) -> int:
    """Replace the table's content with the file's, all or nothing. Returns the entry count."""
    synonyms = read_synonyms(path)
    with connection.transaction():
        connection.execute("DELETE FROM synonym")
        with connection.cursor() as cursor:
            cursor.executemany("INSERT INTO synonym VALUES (%s, %s)", list(synonyms.items()))
    return len(synonyms)
