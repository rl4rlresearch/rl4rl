# Continuous-session Autoresearch protocol

`workshop_primary_block1_continuous_v1.toml` is the current protocol-1.3 launch
preset. It must never be pooled with the ephemeral-session protocol 1.1 or the
full synchronized protocol 1.2 as if their execution plans were identical.

Every run begins one non-ephemeral Codex session at opportunity 1. The runner
stores its thread ID in `runs/<run-id>/state.json` and uses `codex exec resume`
for opportunities 2–200. Codex therefore retains the conversation transcript
within each trajectory. The primary stage contains C0–C3 only; an optional N0
trajectory would also be continuous if activated later.

Before each resumed turn, the controller reconstructs a stable per-run Codex
workspace from that opportunity's selected candidate. The controller then
copies only the editable candidate files back into the immutable,
per-opportunity workspace used for snapshotting and evaluation. The current
filesystem and structured Layer A state remain the authoritative candidate
record, not files remembered from a prior turn.

Consequences for interpretation:

- C0 versus C2 estimates the added effect of controller-provided portfolio
  evidence over a shared continuous conversation, not memory versus no memory.
- N0 is not part of the primary stage. If activated later, it has no
  controller-supplied adaptive state but is not transcript-free.
- C1 and C3 alone receive the identical scheduled assumption-changing text at
  every tenth opportunity: 10, 20, ..., 200.

Protocols 1.2 and 1.3 use an Autoresearch-only intervention template. It asks the
agent to identify a load-bearing assumption from the visible evidence, test a
different computational mechanism, avoid mechanism families already tested
without new evidence, and report the old assumption, alternative mechanism,
and discriminating result. The original transition template remains in force
for the ephemeral and OpenEvolve protocols.

## Scale and expected cost

The protocol-1.3 campaign pre-creates 15 trajectories so all optional extensions
share frozen hashes and block seeds, but the primary launch advances only the
four Block 1 factorial trajectories. It therefore schedules 800 primary Codex
proposals, up to 800 evaluator calls, and 40 scheduled interventions. No N0 or
Block 2/3 call occurs during the primary stage.

The first 14 completed opportunities in the earlier ephemeral C0–C3
Autoresearch campaign averaged about 251,000 accounted tokens, 21.8 minutes
wall time, and 19.7 evaluator minutes per proposal. The historical continuous
Sol xhigh pilots used about 0.50–1.93 million accounted tokens per macro
attempt, including their extra agent interactions and automation polling. The
continuous factorial runner has one bounded Codex turn per proposal, so its
later-turn cost is uncertain; the 500-million-token per-run ceiling is a
capacity limit, not a forecast.

At the current 19.7-minute evaluator mean, the primary stage represents roughly
263 aggregate evaluator-hours. Because its four jobs run concurrently, holding
the early observed rate fixed suggests approximately three to four wall-clock
days, not 19 days. This extrapolation is intentionally conservative: the first
sample consisted of fresh 30,000-step training, whereas later checkpoint-based
or smaller candidates may be faster. Four-way MPS contention may also make each
early evaluation slower than a serial pilot evaluation.

At 251,000 tokens per fresh ephemeral proposal, 800 calls would be about 201
million accounted tokens. Historical continuous-pilot averages imply roughly
0.40–1.54 billion, but include growing transcripts, polling, and automations and
therefore are not a direct forecast. The 500-million-token per-run ceiling is a
capacity limit rather than an intended spend. Measure the first ten primary
waves before projecting the remainder.

Block 1 is the frozen one-block primary analysis, with one trajectory per cell.
It has no between-block replication, so its factorial contrasts are descriptive.
The campaign retains dormant Block 1 N0 and Blocks 2–3. They are valid later
extensions under the same instrument, but an outcome-dependent decision to run
them must be disclosed as adaptive rather than represented as an original
three-block confirmatory design.

Prepare—but do not launch—the campaign with a fresh calibration because this
protocol has a distinct hash:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/workshop_primary_block1_continuous_v1.toml
TASK=$C0C3/configs/tasks/adderboard.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch_continuous.toml
OUT=data/c0c3/workshop-primary-block1-continuous-adderboard-autoresearch

$PY -m $CLI calibrate \
  --protocol "$PROTOCOL" --task "$TASK" \
  --output "$OUT-calibration" --python-bin "$PY"

$PY -m $CLI create \
  --protocol "$PROTOCOL" --task "$TASK" --framework "$FRAMEWORK" \
  --baseline "$OUT-calibration/baseline.json" --output "$OUT-campaign"

$PY -m $CLI validate --campaign "$OUT-campaign"
```

After reviewing the frozen inputs and validation report, launch with
exactly:

```bash
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage factorial
```

This runs 200 concurrent C0–C3 waves and no N0. `status` will continue to show
the eleven dormant extension runs as `ready`; that is expected and consumes no
tokens or evaluator time. Do not use `run-parallel-campaign` or four independent
`run-one` processes.

Only after the primary stage completes, an optional stage can be activated with
the same campaign and hashes, for example:

```bash
# Optional descriptive Block 1 N0
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage no-search

# Optional Block 2 factorial replication
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 2 --stage factorial
```

Timestamp the extension decision and reason before invoking either command.
