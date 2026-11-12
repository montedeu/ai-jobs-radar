"""The MCP server: read-only tools over the radar's database, behind a bearer token."""

import hmac
import os

import psycopg
import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from research_radar.database import connect
from research_radar.history_summary import HistorySummary, history_summary
from research_radar.job_log import JobRun, error_summary, latest_runs

HEALTH_PATH = "/health"

server = MCPServer(
    "ml-research-radar",
    instructions=(
        "Answers which methods, models, tasks and datasets rise and fade in arXiv ML papers. "
        "An independent project, not endorsed by arXiv."
    ),
)


class ServiceStatus(BaseModel):
    latest_runs: list[JobRun]
    # None, with history_error set, when the summary could not be read.
    history: HistorySummary | None = None
    history_error: str | None = None


class StatusUnavailable(Exception):
    """The database cannot be reached. The message is the error type and SQLSTATE only."""


def _type_only(text: str) -> str:
    # "Type: message" and "Type (SQLSTATE x)" both begin with the type.
    return text.split(":", 1)[0]


def _without_error_text(run: JobRun) -> JobRun:
    # The job log may hold full texts (rows written before batch errors were cut down); the
    # answer names only the type.
    details = dict(run.details)
    if isinstance(details.get("error"), str):
        details["error"] = _type_only(details["error"])
    if isinstance(details.get("batch_errors"), dict):
        details["batch_errors"] = {
            batch: _type_only(text) if isinstance(text, str) else "(скрыто)"
            for batch, text in details["batch_errors"].items()
        }
    return run.model_copy(update={"details": details})


@server.tool()
def service_status() -> ServiceStatus:
    """When each scheduled job last ran and how it ended, and what the database holds.

    `history` gives, per slice, the papers per calendar month (by paper date), the backfill
    boundary, the last arXiv update time stored, `collected_through` (every day up to it is
    collected), and the days up to it without a paper. It also gives, once for the whole
    answer, when the daily collection last succeeded and the database size. If the summary
    cannot be read, `history` is empty and `history_error` says why; `latest_runs` stays.
    If the database cannot be reached, the call fails with the error type and SQLSTATE.
    """
    try:
        with connect() as connection:
            runs = [_without_error_text(run) for run in latest_runs(connection)]
            # Ends the transaction the read above began; the summary needs a fresh one.
            connection.commit()
            try:
                return ServiceStatus(latest_runs=runs, history=history_summary(connection))
            except Exception as error:
                return ServiceStatus(latest_runs=runs, history_error=error_summary(error))
    except psycopg.Error as error:
        raise StatusUnavailable(error_summary(error)) from None


@server.custom_route(HEALTH_PATH, methods=["GET"])
async def health(request: Request) -> PlainTextResponse:
    """Up and able to read the migrated database.

    The deploy sends traffic to a new revision on the strength of this answer, so a
    revision that cannot reach its database must not say it is fine.
    """
    try:
        await run_in_threadpool(_read_database)
    except (Exception, SystemExit) as error:  # SystemExit: DATABASE_URL is not set
        return PlainTextResponse(f"database: {type(error).__name__}", status_code=503)
    return PlainTextResponse("ok")


def _read_database() -> None:
    with connect() as connection:
        connection.execute("SELECT 1 FROM job_run LIMIT 1")


class BearerToken:
    """Rejects every request without the expected bearer token, except the health check."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or (
            scope["type"] == "http" and scope["path"] == HEALTH_PATH
        ):
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            # Nothing here speaks websocket; a route added later must not arrive open.
            await send({"type": "websocket.close", "code": 1008})
            return
        given = dict(scope["headers"]).get(b"authorization", b"")
        if not hmac.compare_digest(given, self.expected):
            response = JSONResponse(
                {"error": "missing or wrong bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def application(token: str) -> ASGIApp:
    if not token:
        raise SystemExit("MCP_TOKEN is not set; the server does not start without one")
    # Stateless JSON responses: Cloud Run may stop the single instance between requests,
    # so nothing can rely on an in-memory session. host="0.0.0.0" also turns off the SDK's
    # Host-header allowlist, which only knows localhost; the token is the protection here.
    app = server.streamable_http_app(stateless_http=True, json_response=True, host="0.0.0.0")
    return BearerToken(app, token)


def serve() -> None:
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(application(os.environ.get("MCP_TOKEN", "")), host="0.0.0.0", port=port)
