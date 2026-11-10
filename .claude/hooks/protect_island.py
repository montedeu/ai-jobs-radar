#!/usr/bin/env python3
"""PreToolUse on Edit|MultiEdit|Write|Bash: the growth query is the developer's learning
island (CLAUDE.md). Agents read it, run it and review it; they do not change it.

A text check on a shell command cannot be complete: a script that opens the file from
inside an interpreter passes. The hook stops the direct ways; the rule in CLAUDE.md
holds for the rest.
"""

import json
import re
import subprocess
import sys

ISLAND_NAME = "growth.sql"
ISLAND_PATH = "src/research_radar/sql/growth.sql"
# Matched by name, not by full path: `cd src/research_radar/sql && ... growth.sql` and a
# glob such as sql/g*.sql reach the same file.
NAMES_ISLAND = re.compile(r"growth\.sql|sql/g[^\s/]*\*|\*\.sql")
SEGMENT_BREAK = re.compile(r"&&|\|\||[;|\n]")
REDIRECT_TO_ISLAND = re.compile(r">>?\s*\S*(growth\.sql|\*\.sql|sql/g\S*\*)")
WRITING_COMMAND = re.compile(
    r"^\s*(sudo\s+)?("
    r"tee|rm|mv|cp|truncate|patch|install|dd|ed|ex|"
    r"sed\b.*\s-\w*i|perl\b.*\s-\w*i|"
    r"git\s+(checkout|restore|apply|mv|rm)"
    r")\b"
)
# These need no path to throw away uncommitted work in the whole tree.
TREE_WIDE_REVERT = re.compile(
    r"git\s+(stash(?!\s+(list|show))|reset\s+--hard|clean\b|checkout\s+(--\s+)?\.|restore\s+\.)"
)


def refuse() -> None:
    print(
        f"{ISLAND_PATH} is written by the developer. Describe the problem (place, "
        "consequence, evidence) or add a failing test; do not change the file.",
        file=sys.stderr,
    )
    sys.exit(2)


def island_has_uncommitted_work() -> bool:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ISLAND_PATH], capture_output=True, text=True
    )
    return bool(status.stdout.strip())


def main() -> None:
    tool_input = json.load(sys.stdin).get("tool_input", {})

    if tool_input.get("file_path", "").endswith(ISLAND_NAME):
        refuse()

    command = tool_input.get("command", "")
    for segment in SEGMENT_BREAK.split(command):
        if TREE_WIDE_REVERT.search(segment) and island_has_uncommitted_work():
            refuse()
        if not NAMES_ISLAND.search(segment):
            continue
        # Judged per segment: `cat growth.sql 2>&1`, a `>` inside a commit message, or
        # an `rm` elsewhere on the line are not writes to the island.
        if REDIRECT_TO_ISLAND.search(segment) or WRITING_COMMAND.search(segment):
            refuse()


if __name__ == "__main__":
    main()
