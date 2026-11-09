#!/usr/bin/env bash
# PreToolUse on Edit|Write: the developer writes the Dockerfile and .dockerignore; agents
# report what needs to change instead (issue #12).
f="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')"
case "$(basename "$f")" in
  Dockerfile|.dockerignore)
    echo "$(basename "$f") is written by the developer. Describe the change you need instead of making it." >&2
    exit 2
    ;;
esac
