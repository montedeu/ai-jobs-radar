#!/usr/bin/env bash
# PostToolUse on Bash: lint-on-edit.sh only sees files written through Edit|Write, and a
# file written by a shell command (a heredoc, a script) would otherwise wait for CI.
# Checks only; it never rewrites a file behind the agent.
command="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))')"
# Cheap gate: most commands touch no Python file, and uv run is not free.
[[ "$command" == *.py* ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
mapfile -t files < <({ git diff --name-only --diff-filter=d HEAD; git ls-files --others --exclude-standard; } 2>/dev/null | grep '\.py$' | sort -u)
(( ${#files[@]} )) || exit 0
if ! out="$(uv run --quiet ruff check --quiet --output-format concise "${files[@]}" 2>&1 && uv run --quiet ruff format --check --quiet "${files[@]}" 2>&1)"; then
  printf 'Lint on files changed in the working tree:\n%s\nFix: uv run ruff check --fix . && uv run ruff format .\n' "$out" >&2
  exit 2
fi
