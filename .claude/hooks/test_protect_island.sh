#!/usr/bin/env bash
# Cases for protect_island.py: each line is the expected exit code, a tool-input field and
# its value. Run from the repo root: .claude/hooks/test_protect_island.sh
cd "$(dirname "$0")/../.." || exit 1
I=src/research_radar/sql/growth.sql
failed=0

check() {
  local want="$1" field="$2" value="$3" got
  python3 -c 'import json,sys; print(json.dumps({"tool_input": {sys.argv[1]: sys.argv[2]}}))' "$field" "$value" \
    | .claude/hooks/protect_island.py >/dev/null 2>&1
  got=$?
  if [[ "$got" != "$want" ]]; then
    printf 'FAIL want %s got %s: %s\n' "$want" "$got" "$value"
    failed=1
  fi
}

# Writes: blocked.
check 2 file_path "/repo/$I"
check 2 command "echo x > $I"
check 2 command "echo x >> $I"
check 2 command "cd src/research_radar/sql && echo x >> growth.sql"
check 2 command "sed -i s/a/b/ $I"
check 2 command "perl -pi -e s/a/b/ $I"
check 2 command "cp /tmp/x src/research_radar/sql/g*.sql"
check 2 command "git checkout main -- $I"
check 2 command "git restore $I"
check 2 command "rm $I"
check 2 command "uv run pytest -q && mv /tmp/x $I"
check 2 command "cat /tmp/x | tee $I"

# Reads and unrelated work: allowed.
check 0 file_path "/repo/src/research_radar/growth.py"
check 0 command "cat $I"
check 0 command "cat $I 2>&1"
check 0 command "cat $I > /tmp/copy.sql"
check 0 command "psql -f $I"
check 0 command "uv run pytest tests/test_growth.py"
check 0 command "git commit -m 'a -> b, see $I'"
check 0 command "git log -p -- $I | wc -l; echo mv"
check 0 command "uv run pytest tests/test_growth.py; rm -rf tmp; cat $I"
check 0 command "git diff $I | head; git restore --staged README.md"
check 0 command "echo x > README.md"

if (( failed )); then exit 1; fi
echo "all cases pass"
