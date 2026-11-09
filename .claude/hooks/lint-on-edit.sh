#!/usr/bin/env bash
# PostToolUse on Edit|Write: report lint and format findings on the edited file at once,
# so they don't wait for CI. Checks only; it never rewrites the file behind the agent.
f="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')"
[[ -f "$f" ]] || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
case "$f" in
  *.py)
    if ! out="$(uv run --quiet ruff check --quiet --output-format concise "$f" 2>&1 && uv run --quiet ruff format --check --quiet "$f" 2>&1)"; then
      printf 'Lint on %s:\n%s\nFix: uv run ruff check --fix %s && uv run ruff format %s\n' "$f" "$out" "$f" "$f" >&2
      exit 2
    fi
    ;;
esac
