#!/usr/bin/env bash
set -euo pipefail

CODEX_MODEL=gpt-5.6-sol
CODEX_REASONING_EFFORT=xhigh
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$RUN_DIR/workspace"
mkdir -p "$RUN_DIR/logs"

# This starts a real Codex agent session. It can consume your configured Codex
# or API allowance; inspect RUN_CONFIG.json before launching.
codex exec --model "$CODEX_MODEL" \
  -c "model_reasoning_effort=$CODEX_REASONING_EFFORT" \
  --json --approve-for-me \
  --cd "$WORKSPACE" --add-dir "$RUN_DIR" \
  --output-last-message "$RUN_DIR/logs/codex-last-message.md" \
  "$(cat "$RUN_DIR/AGENT_PROMPT.md")" \
  | tee "$RUN_DIR/logs/codex-events.jsonl"
