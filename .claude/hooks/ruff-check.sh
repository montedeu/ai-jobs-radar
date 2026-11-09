#!/usr/bin/env bash
# PostToolUse on Edit|Write: report ruff findings on the edited Python file at once, so
# they don't wait for CI. Checks only; it never rewrites the file behind the agent's back.
f="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')"
[[ "$f" == *.py && -f "$f" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
if ! out="$(uv run --quiet ruff check --quiet --output-format concise "$f" 2>&1 && uv run --quiet ruff format --check --quiet "$f" 2>&1)"; then
  printf 'ruff on %s:\n%s\nFix: uv run ruff check --fix %s && uv run ruff format %s\n' "$f" "$out" "$f" "$f" >&2
  exit 2
fi
