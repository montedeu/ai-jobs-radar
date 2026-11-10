#!/usr/bin/env bash
# PreToolUse on Edit|Write|Bash: the growth query is the developer's learning island
# (CLAUDE.md). Agents read it, run it and review it; they do not change it.
ISLAND="sql/growth.sql"
input="$(cat)"
field() { printf '%s' "$input" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input', {}).get('$1', ''))"; }

refuse() {
  echo "src/research_radar/$ISLAND is written by the developer. Describe the problem (place, consequence, evidence) or add a failing test; do not change the file." >&2
  exit 2
}

case "$(field file_path)" in
  *"$ISLAND") refuse ;;
esac

command="$(field command)"
if [[ "$command" == *"$ISLAND"* ]]; then
  # Reading the file (cat, EXPLAIN through psql, pytest) stays allowed; anything that
  # can write, move or restore it does not.
  if grep -Eq '>|\btee\b|\bsed\b[^|;]*-i|\b(rm|mv|cp|truncate|patch|install|dd)\b|git +(checkout|restore|apply|stash|reset)' <<<"$command"; then
    refuse
  fi
fi
exit 0
