"""Connections to the radar's Postgres database."""

import os
import time

import psycopg

# Neon's free plan suspends the database after five idle minutes and takes a few seconds
# to wake; the first connection of the day would otherwise fail the whole job or request.
CONNECT_ATTEMPTS = 5
CONNECT_TIMEOUT_SECONDS = 15


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    return url


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
