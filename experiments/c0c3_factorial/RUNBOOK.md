# C0–C3 runbook

Run every command from the repository root. Paths below are repository-relative
or operator-chosen; no user-specific absolute path is embedded in a campaign.

## 1. Choose immutable inputs

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/dev.toml
TASK=$C0C3/configs/tasks/adderboard.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch.toml
OUT=data/c0c3/dev-adderboard-autoresearch
```

Use `dev.toml` until the entire workflow succeeds. Change only `PROTOCOL` to
`paper_v1.toml` for the frozen paper campaign. For OpenEvolve, change only:

```bash
FRAMEWORK=$C0C3/configs/frameworks/openevolve.toml
```

For the separately labeled local parallel workshop pilot, use:

```bash
PROTOCOL=$C0C3/configs/protocols/workshop_pilot_parallel_v1.toml
TASK=$C0C3/configs/tasks/adderboard.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch.toml
OUT=data/c0c3/workshop-pilot-parallel-adderboard-autoresearch
```

Do not reuse a `dev.toml` or `paper_v1.toml` calibration/campaign: protocol 1.1
has a different protocol hash. The parallel protocol currently requires a local
task backend and is not a Modal launch path.

Verify prerequisites:

```bash
codex --version
$PY -c 'import torch; print(torch.__version__)'
$PY -m pytest -q tests/test_c0c3_factorial_core.py tests/test_c0c3_execution.py
```

Codex must already be authenticated. Scientific calls use the model and
reasoning effort frozen in the protocol TOML, not a user config default.

## 2. Calibrate the frozen seed on the target backend

Calibration output directories must not already exist.

```bash
$PY -m $CLI calibrate \
  --protocol "$PROTOCOL" \
  --task "$TASK" \
  --output "$OUT-calibration" \
  --python-bin "$PY"
```

Expected final artifact:

```text
$OUT-calibration/baseline.json
```

Open it and confirm `calibration_kind` is
`executed_on_target_backend`, the task is correct, and metrics are plausible.
AdderBoard must qualify at or above 0.99 accuracy.

One calibration can be reused for both framework campaigns only when protocol,
task source, task configuration, and backend are identical. A protocol change
changes the protocol hash and deliberately requires a new calibration.

For a GPU task that cannot run locally, use the prepare/execute split described
in [MODAL.md](MODAL.md). Do not calibrate nanoGPT on one GPU type and run its
candidates on another: its fixed-time objective is hardware-specific.

## 3. Create a campaign

```bash
$PY -m $CLI create \
  --protocol "$PROTOCOL" \
  --task "$TASK" \
  --framework "$FRAMEWORK" \
  --baseline "$OUT-calibration/baseline.json" \
  --output "$OUT-campaign"
```

By default this creates C0–C3 plus N0 in every block. `--without-no-search` is
available only for engineering diagnostics; paper campaigns include N0.

Campaign contents include:

```text
campaign.json                 immutable hashes and identifiers
inputs/                       frozen protocol/task/framework JSON
schedule.json                 block, order, condition, run seed, run ID
runs/<run-id>/manifest.json   assignment and runtime hash
runs/<run-id>/state.json      crash-safe mutable controller state
runs/<run-id>/events.jsonl    append-only scientific event log
runs/<run-id>/task-support/   copied evaluator/task source
runs/<run-id>/candidates/     content-addressed editable snapshots
```

Do not edit these files by hand.

## 4. Run the fail-closed launch audit

```bash
$PY -m $CLI validate --campaign "$OUT-campaign"
```

The command must exit zero and print `"valid": true`. It checks:

- exactly one C0–C3 assignment per block and paired block seed;
- contiguous frozen order and unique run IDs;
- byte-identical starting candidates and task-support trees;
- protocol, task, framework, and scientific-runtime hashes;
- untouched ready state for every run;
- every opportunity’s treatment-redacted prompt skeleton;
- correct factor pairing across C0/C1/C2/C3;
- absence of Layer B/C artifacts.

Validation is a launch gate, not a general health command. It intentionally
fails after a run begins. Use `status` after launch.

Archive the printed `validation.json`, `codex --version`, Python environment
lock/pip freeze, GPU model/driver, OS, git commit, and the campaign hashes before
starting the first paper opportunity.

## 5. Execute with the protocol's frozen rule

### Protocol 1.0 serial blocked round-robin

Run exactly one next opportunity:

```bash
$PY -m $CLI run-next \
  --campaign "$OUT-campaign" \
  --python-bin "$PY"
```

Run a bounded batch in blocked round-robin order:

```bash
$PY -m $CLI run-campaign \
  --campaign "$OUT-campaign" \
  --python-bin "$PY" \
  --max-opportunities 10
```

Run until the campaign completes by omitting `--max-opportunities`:

```bash
$PY -m $CLI run-campaign \
  --campaign "$OUT-campaign" \
  --python-bin "$PY"
```

`run-one --run-id` and `run --run-id` are diagnostic escape hatches. They do not
enforce campaign ordering and must not be used for a confirmatory campaign.

A file lock prevents two local processes from mutating one run, but it is not
authorization to run multiple writers against one campaign. Keep exactly one
campaign orchestrator active.

### Protocol 1.1 synchronized parallel condition rounds

Run one block wave. Normally this launches C0–C3 concurrently and then N0
serially, so one wave ordinarily consumes five Codex calls:

```bash
$PY -m $CLI run-parallel-next \
  --campaign "$OUT-campaign" \
  --python-bin "$PY"
```

Run a bounded number of block waves:

```bash
$PY -m $CLI run-parallel-campaign \
  --campaign "$OUT-campaign" \
  --python-bin "$PY" \
  --max-block-rounds 3
```

Omit `--max-block-rounds` to finish the campaign. With three blocks and 30
opportunities, a clean complete campaign has 90 block waves, 360 factorial
calls, and 90 serial N0 calls. The runner chooses the campaign-wide
least-advanced opportunity count and earliest eligible block; do not manually
choose four run IDs or start four CLI processes.

The outer CLI is the only campaign writer. It owns one campaign lock while an
internal four-worker pool runs distinct run directories. It waits for all
factorial workers before starting N0. Every wave is recorded in
`parallel-rounds.jsonl`. A nonzero command exit may leave active opportunities;
use the recovery procedure below before invoking the parallel command again.

## 6. Inspect progress without changing it

```bash
$PY -m $CLI status --campaign "$OUT-campaign"
```

Each tab-separated row is:

```text
run_id  condition  status  proposals_used  evaluations_used  total_tokens  evaluator_seconds_remaining
```

For detailed provenance, inspect `events.jsonl` and individual
`opportunities/NNNN/` directories. Never edit them. The last
`proposal_completed` event is authoritative; a Codex last message alone is not.
For protocol 1.1, also inspect `parallel-rounds.jsonl` for the exact concurrent
participant set, N0 member, and `recovery_subset` flag.

## 7. Recover an interrupted active opportunity

If `run-next` reports an active opportunity, first inspect its opportunity
directory, Codex JSONL, evaluator logs, host/process state, and the reason the
process disappeared. Once it is certain no writer is alive:

```bash
$PY -m $CLI recover-active \
  --campaign "$OUT-campaign" \
  --run-id '<exact-run-id>' \
  --reason 'host process terminated after opportunity start'
```

Recovery consumes that proposal, recovers logged Codex tokens when possible,
records no evaluator call unless one had already been durably completed by the
normal runner, and never retries or deletes artifacts. Do not use recovery to
erase a scientifically inconvenient result.

For protocol 1.1, repeat `recover-active` for every run whose state is active.
The next `run-parallel-next` deterministically selects only any still-lagging
factorial peers, marks that wave as a recovery subset, and runs N0 afterward if
it is at the same minimum. Never repeat a peer that already completed.

## 8. Export and adjudicate Layer B

Only after `status` shows every run completed:

```bash
$PY -m $CLI export-layer-b --campaign "$OUT-campaign"
```

This creates opaque, randomly ordered parent/candidate packets and a private
condition mapping. Do not show reviewers `sealed-layer-b/private/`, event logs,
run directories, or Layer A scores.

Recommended review process:

1. Give two reviewers separate copies of `annotations.template.tsv` and the
   packet directories in `packet_order.tsv` order.
2. Require one row per packet using the rubric in `PAPER_NOTES.md`.
3. Compute/report raw qualification agreement and cluster agreement.
4. Adjudicate disagreements while still blinded.
5. Freeze one final `annotations.adjudicated.tsv`.

Score once:

```bash
$PY -m $CLI score-layer-b \
  --campaign "$OUT-campaign" \
  --annotations "$OUT-campaign/sealed-layer-b/annotations.adjudicated.tsv"
```

Outputs:

- `factorial_outcomes.tsv` — C0–C3 run outcomes;
- `no_search_outcomes.tsv` — N0 only;
- `factorial_estimates.json` — cell means, main effects, interaction, and
  within-block contrasts.

`score-layer-b` refuses duplicate, missing, or extra packet IDs and refuses a
qualified row without a cluster label.

## 9. Run sealed Layer C

Layer C is independent of Layer B adjudication but still requires all runs to
be completed:

```bash
$PY -m $CLI run-layer-c \
  --campaign "$OUT-campaign" \
  --python-bin "$PY"
```

For AdderBoard this uses the disjoint fixed seed. For official Karpathy
Autoresearch it is a repeat of the pinned validation procedure and must be
reported as replication, not unseen generalization.

## 10. Run all task/framework strata

The minimum planned matrix consists of four independent campaigns:

| Task | Framework configuration |
|---|---|
| AdderBoard | `autoresearch.toml` |
| AdderBoard | `openevolve.toml` |
| Karpathy nanoGPT | `autoresearch.toml` |
| Karpathy nanoGPT | `openevolve.toml` |

Paper v1 creates 15 runs × 100 proposals = 1,500 proposals per campaign, or
6,000 proposals across the four strata. Confirm compute, Codex allowance, Modal
budget, storage, and reviewer capacity before launching. If resources force a
smaller design, preregister the changed block/proposal counts as a new protocol;
do not quietly stop paper v1 early.

## 11. Final archive

Preserve, read-only:

- the complete campaign directory;
- calibration bundle and baseline;
- source commit and dependency locks;
- launch validation receipt and environment receipt;
- all Codex/evaluator logs and candidate snapshots;
- independent reviewer files and adjudicated file;
- Layer B/C outputs and analysis scripts;
- every protocol deviation with timestamp and affected run IDs.

Create hashes for the archive before transferring it. Never publish secrets,
Codex authentication state, or the Layer B private salt/mapping before review is
frozen.
