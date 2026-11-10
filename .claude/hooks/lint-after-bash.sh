#!/usr/bin/env bash
# PostToolUse on Bash: lint-on-edit.sh only sees files written through Edit|Write, and a
# file written by a shell command (a heredoc, a script) would otherwise wait for CI.
# Checks only; it never rewrites a file behind the agent.
command="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))')"
# Cheap gate: most commands touch no Python file, and uv run is not free.
grep -Eq '\.py([^a-z_]|$)' <<<"$command" || exit 0
command -v uv >/dev/null || exit 0
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# Only files written in the last two minutes: an older lint error in the working tree is
# not this command's doing, and reporting it would block every later command.
mapfile -d '' -t changed < <({ git diff -z --name-only --diff-filter=d HEAD; git ls-files -z --others --exclude-standard; } 2>/dev/null)
files=()
now=$(date +%s)
for f in "${changed[@]}"; do
  [[ "$f" == *.py && -f "$f" ]] || continue
  (( now - $(stat -c %Y "$f") <= 120 )) && files+=("$f")
done
(( ${#files[@]} )) || exit 0

if ! out="$(uv run --quiet ruff check --quiet --output-format concise "${files[@]}" 2>&1 && uv run --quiet ruff format --check --quiet "${files[@]}" 2>&1)"; then
  printf 'Lint on files this command changed:\n%s\nFix: uv run ruff check --fix . && uv run ruff format .\n' "$out" >&2
  exit 2
fi
