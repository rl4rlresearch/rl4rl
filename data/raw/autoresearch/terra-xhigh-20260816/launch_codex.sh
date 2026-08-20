#!/usr/bin/env bash
set -uo pipefail

CODEX_MODEL=gpt-5.6-terra
CODEX_REASONING_EFFORT=xhigh
PYTHON_BIN=/Users/utshaho/Documents/GitHub/rl4rl/architecture_discovery/.venv/bin/python
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$RUN_DIR/workspace"
mkdir -p "$RUN_DIR/logs"
LAUNCH_LOCK="$RUN_DIR/.launcher.lock"

if ! mkdir "$LAUNCH_LOCK" 2>/dev/null; then
  owner="$(cat "$LAUNCH_LOCK/pid" 2>/dev/null || true)"
  if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
    echo "A launcher is already active for this run (pid $owner)." >&2
    exit 3
  fi
  rm -rf "$LAUNCH_LOCK"
  mkdir "$LAUNCH_LOCK"
fi
printf '%s
' "$$" > "$LAUNCH_LOCK/pid"
trap 'rm -rf "$LAUNCH_LOCK"' EXIT INT TERM

read_json_int() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' "$1" "$2"
}

MAX_ATTEMPTS="$(read_json_int "$RUN_DIR/RUN_CONFIG.json" max_attempts)"
session=0

while [ "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" -lt "$MAX_ATTEMPTS" ]; do
  session=$((session + 1))
  printf '%s session=%s attempts=%s/%s starting Codex
' "$(date -u +%FT%TZ)" "$session" "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
  codex exec --model "$CODEX_MODEL" \
    -c "model_reasoning_effort=$CODEX_REASONING_EFFORT" \
    --json --approve-for-me \
    --cd "$WORKSPACE" --add-dir "$RUN_DIR" \
    --output-last-message "$RUN_DIR/logs/codex-last-message.md" \
    "$(cat "$RUN_DIR/AGENT_PROMPT.md")" \
    | tee -a "$RUN_DIR/logs/codex-events.jsonl"
  codex_exit=${PIPESTATUS[0]}
  used="$(read_json_int "$RUN_DIR/STATE.json" attempts_used)"
  if [ "$used" -lt "$MAX_ATTEMPTS" ]; then
    printf '%s session=%s exited=%s before budget (%s/%s); relaunching in 10s
' "$(date -u +%FT%TZ)" "$session" "$codex_exit" "$used" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
    sleep 10
  fi
done
printf '%s budget exhausted at %s/%s; supervisor exiting
' "$(date -u +%FT%TZ)" "$(read_json_int "$RUN_DIR/STATE.json" attempts_used)" "$MAX_ATTEMPTS" >> "$RUN_DIR/logs/supervisor.log"
