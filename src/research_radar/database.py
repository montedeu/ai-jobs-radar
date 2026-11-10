"""Connections to the radar's Postgres database."""

import os
import re
import time

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

# Neon's free plan suspends the database after five idle minutes and takes a few seconds
# to wake; the first connection of the day would otherwise fail the whole job or request.
CONNECT_ATTEMPTS = 5
CONNECT_TIMEOUT_SECONDS = 15


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    return url


def direct_url(url: str) -> str:
    """The same database reached without Neon's connection pooler.

    Neon's pooled address is the direct one with "-pooler" after the endpoint id, and
    its console offers the pooled address first. A job holds a session-level lock for
    its whole run, and behind the pooler a session is not the job's own.
    """
    parts = conninfo_to_dict(url)
    hosts = [_without_pooler(host) for host in (parts.get("host") or "").split(",")]
    if parts.get("hostaddr") and hosts != (parts.get("host") or "").split(","):
        # libpq connects to hostaddr and ignores the name, so renaming would change nothing.
        raise ValueError("the database address pins a pooler by hostaddr; use the direct URL")
    if parts.get("host"):
        parts["host"] = ",".join(hosts)
    if parts.get("options"):
        # Clients without SNI name the endpoint here instead of in the host.
        parts["options"] = _POOLED_ENDPOINT_OPTION.sub(r"\1", parts["options"])
    return make_conninfo(**parts)


_POOLED_ENDPOINT_OPTION = re.compile(r"(endpoint=[\w-]+?)-pooler\b", re.IGNORECASE)


def _without_pooler(host: str) -> str:
    endpoint, dot, rest = host.partition(".")
    if endpoint.lower().endswith("-pooler"):
        endpoint = endpoint[: -len("-pooler")]
    return endpoint + dot + rest


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or database_url()
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS)
        except psycopg.OperationalError:
            if attempt == CONNECT_ATTEMPTS:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")
