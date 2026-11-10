"""The daily collection job. For now it only proves the scheduled path to the database."""

from research_radar.database import connect
from research_radar.job_log import record_run

JOB = "collect"


def run(url: str | None = None) -> None:
    with connect(url) as connection, record_run(connection, JOB) as details:
        details["papers_received"] = 0
        details["note"] = "skeleton: nothing is collected yet"
