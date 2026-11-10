"""The MCP server: read-only tools over the radar's database, behind a bearer token."""

import hmac
import os

import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from research_radar.database import connect
from research_radar.job_log import JobRun, latest_runs

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


@server.tool()
def service_status() -> ServiceStatus:
    """When each of the radar's scheduled jobs last ran and how it ended."""
    with connect() as connection:
        return ServiceStatus(latest_runs=latest_runs(connection))


@server.custom_route(HEALTH_PATH, methods=["GET"])
async def health(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


class BearerToken:
    """Rejects every request without the expected bearer token, except the health check."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] == HEALTH_PATH:
            await self.app(scope, receive, send)
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
