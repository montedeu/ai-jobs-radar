FROM ghcr.io/astral-sh/uv:0.11.33 AS uv

# --- build: resolve and install everything into /app/.venv ---
FROM python:3.12.16-slim-trixie AS build

COPY --from=uv /uv /bin/

# Compile .pyc at build time so a cold start doesn't pay for it; copy from the cache
# mount instead of hard-linking (it's a different filesystem); never download a Python,
# the venv must point at this image's interpreter, which the runtime stage shares.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-editable

# --- runtime: only the venv, no uv, no source ---
FROM python:3.12.16-slim-trixie

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

COPY --from=build /app/.venv .venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# One image for every Cloud Run resource; each overrides the arguments (migrate, collect).
# Exec form: radar is PID 1 and gets Cloud Run's SIGTERM.
ENTRYPOINT ["radar"]
CMD ["serve-mcp"]
