# Continuous-session Autoresearch protocol

`workshop_pilot_parallel_continuous_v1.toml` is a separately labeled protocol
stratum. It must never be pooled with the ephemeral-session protocol 1.1.

Every run begins one non-ephemeral Codex session at opportunity 1. The runner
stores its thread ID in `runs/<run-id>/state.json` and uses `codex exec resume`
for opportunities 2–30. Codex therefore retains the conversation transcript in
all five conditions, including C0, C2, and N0.

Before each resumed turn, the controller reconstructs a stable per-run Codex
workspace from that opportunity's selected candidate. The controller then
copies only the editable candidate files back into the immutable,
per-opportunity workspace used for snapshotting and evaluation. The current
filesystem and structured Layer A state remain the authoritative candidate
record, not files remembered from a prior turn.

Consequences for interpretation:

- C0 versus C2 estimates the added effect of controller-provided portfolio
  evidence over a shared continuous conversation, not memory versus no memory.
- N0 has no controller-supplied adaptive state, but is not transcript-free.
- C1 and C3 alone receive the identical scheduled assumption-changing text at
  opportunities 10 and 20.

Prepare—but do not launch—the campaign with a fresh calibration because this
protocol has a distinct hash:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/workshop_pilot_parallel_continuous_v1.toml
TASK=$C0C3/configs/tasks/adderboard.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch_continuous.toml
OUT=data/c0c3/workshop-pilot-parallel-continuous-adderboard-autoresearch

$PY -m $CLI calibrate \
  --protocol "$PROTOCOL" --task "$TASK" \
  --output "$OUT-calibration" --python-bin "$PY"

$PY -m $CLI create \
  --protocol "$PROTOCOL" --task "$TASK" --framework "$FRAMEWORK" \
  --baseline "$OUT-calibration/baseline.json" --output "$OUT-campaign"

$PY -m $CLI validate --campaign "$OUT-campaign"
```

After reviewing the frozen inputs and validation report, launch with
`run-parallel-campaign`; do not use four independent `run-one` processes.
