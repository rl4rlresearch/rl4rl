# Continuous-session Autoresearch protocol

`workshop_pilot_parallel_continuous_v1.toml` is a separately labeled protocol
stratum. It must never be pooled with the ephemeral-session protocol 1.1.

Every run begins one non-ephemeral Codex session at opportunity 1. The runner
stores its thread ID in `runs/<run-id>/state.json` and uses `codex exec resume`
for opportunities 2–200. Codex therefore retains the conversation transcript in
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
  every tenth opportunity: 10, 20, ..., 200.

Protocol 1.2 uses an Autoresearch-only intervention template. It asks the
agent to identify a load-bearing assumption from the visible evidence, test a
different computational mechanism, avoid mechanism families already tested
without new evidence, and report the old assumption, alternative mechanism,
and discriminating result. The original transition template remains in force
for the ephemeral and OpenEvolve protocols.

## Scale and expected cost

The protocol contains 15 trajectories: C0–C3 plus N0 in each of three blocks.
At 200 opportunities per trajectory it therefore schedules 3,000 Codex
proposals and up to 3,000 evaluator calls. This is not comparable to one
200-attempt pilot run.

The first 14 completed opportunities in the earlier ephemeral C0–C3
Autoresearch campaign averaged about 251,000 accounted tokens, 21.8 minutes
wall time, and 19.7 evaluator minutes per proposal. The historical continuous
Sol xhigh pilots used about 0.50–1.93 million accounted tokens per macro
attempt, including their extra agent interactions and automation polling. The
continuous factorial runner has one bounded Codex turn per proposal, so its
later-turn cost is uncertain; the 500-million-token per-run ceiling is a
capacity limit, not a forecast.

At the two completed first-wave timings observed before this protocol was
created, one five-run block wave averaged 45.9 minutes. Holding that rate fixed
would put all 600 block waves near 459 wall-clock hours (about 19 days). Smaller
later candidates may evaluate faster, but this remains a multi-day campaign
with approximately 983 aggregate evaluator-hours at the current per-proposal
mean. Record actual rates during calibration and early waves rather than
treating these projections as guarantees.

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
