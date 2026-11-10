"""Checks a deployed MCP server from outside: it is up, and it refuses callers without a token.

Usage: python3 scripts/smoke_test.py https://radar-mcp-....run.app

Standard library only: the deploy workflow runs it without installing the project. It holds
no token, so it proves the protection rather than a tool call.
"""

import sys
import urllib.error
import urllib.request

# A sleeping Cloud Run instance has to start before it answers.
TIMEOUT_SECONDS = 60


def status(url: str, data: bytes | None = None) -> int:
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def main() -> None:
    base = sys.argv[1].rstrip("/")
    checks = {
        "GET /health answers 200": status(f"{base}/health") == 200,
        "POST /mcp without a token answers 401": status(f"{base}/mcp", data=b"{}") == 401,
    }
    for name, passed in checks.items():
        print(("ok    " if passed else "FAILED") + " " + name)
    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
