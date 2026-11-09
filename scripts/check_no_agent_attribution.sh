#!/usr/bin/env bash
# Fails when a commit message or PR body credits an AI agent (Co-Authored-By: Claude,
# "Generated with Claude Code"). Commits here carry the developer's name only.
#
#   check_no_agent_attribution.sh <file>               a message file (the commit-msg hook)
#   check_no_agent_attribution.sh --range <base>..<head>  every commit in the range (CI)
#   check_no_agent_attribution.sh --stdin              text on stdin, e.g. a PR body (CI)
set -euo pipefail

# Anchored to the start of a line, where trailers and footers go, so prose that quotes
# them passes.
pattern='^[[:space:]]*co-authored-by:.*(claude|anthropic)|^[^[:alnum:]]*generated (with|by) \[?claude'

case "${1:-}" in
  --range)
    found=0
    shas="$(git rev-list "$2")"  # a bad range fails here, under set -e
    for sha in $shas; do
      if git log -1 --format=%B "$sha" | grep -qiE "$pattern"; then
        echo "Commit $(git log -1 --format='%h %s' "$sha") credits an AI agent:" >&2
        git log -1 --format=%B "$sha" | grep -iE "$pattern" | sed 's/^/  /' >&2
        found=1
      fi
    done
    exit "$found"
    ;;
  --stdin) text="$(cat)" ;;
  "") echo "usage: $0 <message-file> | --range <base>..<head> | --stdin" >&2; exit 2 ;;
  *) text="$(cat "$1")" ;;
esac

if grep -iE "$pattern" <<<"$text" >&2; then
  echo "The lines above credit an AI agent; remove them." >&2
  exit 1
fi
