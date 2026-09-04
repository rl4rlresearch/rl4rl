# Handoff: controlled C0–C3 research experiments

This document describes the current implementation, with historical
provenance included only to explain the relationship between
`architecture_discovery/` and `experiments/`.

## 1. Current state and provenance

- Historical source context: `codex/c0c3-codex-factorial`
- Historical starting-point commit: `4517f548fddfba132814f2e64b78e862f376a27d`
- Historical starting-point commit: `Merge pull request #4 from rl4rlresearch/autoresearch-pilot`
- Historical implementation snapshot: `3d446168874fa2eeb21ae5c3489367b79661b426`
- Purpose: provide a controlled, reproducible factorial experiment
  comparing research-agent search-state and proposal-policy interventions.

The implementation is present under `experiments/c0c3_factorial/`. The
historical comparison found no tracked changes under `architecture_discovery/`:

```text
git diff 4517f54..HEAD -- architecture_discovery
<empty>
```

`architecture_discovery/` remains the broader inherited
architecture-search infrastructure. The C0–C3 harness is a
framework-neutral experiment package in `experiments/`.

The working tree may contain local campaign outputs under `data/c0c3/` and a
dirty OpenEvolve vendor submodule. Those are runtime artifacts and are not part
of the tracked implementation. Do not commit or delete them without an
explicit data-management decision.

## 2. What existed before this implementation

At the historical starting snapshot, `architecture_discovery/` already contained a large
architecture-discovery system, including:

- generic, semantic, and greedy Autoresearch/OpenEvolve agent entry points;
- Modal launch and boundary code;
- architecture-IR graph, codec, interpreter, and runtime-evidence modules;
- candidate, artifact, lineage, evaluator, trainer, and training-data
  infrastructure;
- research-ledger, replication, review, blinding, sealed-evaluation, novelty,
  and reporting modules;
- local/MPS and Modal readiness scripts and engineering pilots;
- inherited OpenEvolve adapters, policies, process controls, and provider
  attempt accounting.

That system is useful background and contains reusable task/evaluation pieces,
but it does not itself implement the frozen C0–C3 factorial controller. The
current implementation leaves the architecture-discovery engine in place.

## 3. Historical implementation sequence

The implementation was built in these stages:

1. `53da21c Add frozen C0-C3 factorial core`
   - Added strict factorial specifications, C0–C3 condition mapping, state
     controller, budgets, retention rules, deterministic assignments, and
     prompt treatment slots.
2. `8aa5adf Add Codex Autoresearch and OpenEvolve runners`
   - Added non-interactive Codex transport, direct Autoresearch editing,
     Greedy OpenEvolve SEARCH/REPLACE editing, task evaluation, campaign
     construction, and one-opportunity execution.
3. `ae5e909 Harden C0-C3 execution and sealed analysis`
   - Added validation, crash-safe accounting, failure/recovery behavior,
     sealed Layer B packet export/scoring, Layer C evaluation, and analysis.
4. `93d5be5 Add serialized Modal campaign execution`
   - Added the Modal preparation/execution path, persistent volume handling,
     remote campaign path confinement, and serialized GPU execution.
5. `7415fc0 Add portable calibration and blinded review`
   - Added target-backend calibration preparation/execution, portable baseline
     artifacts, blinded review support, and provenance checks.
6. `ca6fe9a Document and freeze the C0-C3 paper protocol`
   - Added the protocol, runbook, framework/task documentation, paper notes,
     and confirmatory protocol configuration.
7. `b54d01b Record C0-C3 completion audit`
   - Added the implementation-completeness and launch-boundary audit.
8. `f7ecd75 Add parallel C0-C3 workshop protocol`
   - Added protocol 1.1, synchronized parallel C0–C3 waves, serial N0 calls,
     campaign locking, recovery-subset handling, and parallel-wave logging.
9. `153ad93 Harden C0-C3 Codex launch compatibility`
   - Hardened non-interactive Codex CLI invocation and compatibility behavior.
10. `3d44616 Preserve virtualenv interpreter in C0-C3 evaluator`
    - Preserved the configured Python interpreter when evaluator commands run,
      including symlinked virtualenv interpreters.

## 4. Current `experiments/` layout

### `experiments/c0c3_factorial/spec.py`

The immutable contract layer:

- `FactorialSpec` loads strict TOML protocol files and computes a canonical
  protocol hash.
- `Condition` defines the four cells:
  - C0: single incumbent, ordinary proposals;
  - C1: single incumbent, scheduled assumption-changing proposals;
  - C2: portfolio memory, ordinary proposals;
  - C3: portfolio memory, scheduled assumption-changing proposals.
- `ModelSpec`, `BudgetSpec`, `TaskSpec`, and `FrameworkSpec` freeze model,
  budget, task, and adapter details.
- `make_assignments()` deterministically shuffles C0–C3 within each block from
  the study seed; all four cells in a block share that block's run seed.
- Protocol 1.0 selects serial blocked execution.
- Protocol 1.1 selects synchronized parallel C0–C3 waves.

### `state.py`

`SearchController` is the single source of truth for one run. It owns:

- immutable seed and candidate records;
- active-opportunity transitions;
- opportunity, evaluation, evaluator-time, and token budgets;
- single-incumbent and portfolio retention;
- deterministic parent selection;
- append-only scientific events and atomic `state.json` writes;
- recovery of interrupted active opportunities.

Portfolio capacity is frozen as `K=4` in the paper/workshop protocols. The
portfolio fills open slots first; after filling, a candidate replaces its
selected parent only on strict fitness improvement. The globally best retained
candidate is tracked as the incumbent, but is not necessarily the next parent.

### `prompts.py` and `templates/`

Prompts are assembled from one common template with exactly two auditable
treatment slots:

- search-state slot: single incumbent versus portfolio state;
- proposal-policy slot: ordinary versus scheduled assumption-changing policy.

The renderer hashes the common template, search-state text, policy text, and
full prompt. Layer-A-visible metrics and candidates are inserted into every
proposal prompt. Layer B and Layer C information is explicitly excluded.

The assumption-changing template requires the agent to challenge a shared core
assumption and propose a meaningfully different architecture family. A simple
width/depth tweak, deletion, scalar adjustment, or renamed version of the same
computation does not satisfy the instruction.

### `codex_cli.py`

`CodexCli` invokes the installed Codex CLI non-interactively with JSONL event
logging, a captured last message, timeout handling, and final-turn token
accounting. Existing protocols use fresh ephemeral Codex calls.

### `frameworks.py`

Two controlled proposal adapters are implemented:

- `AutoresearchAdapter`: Codex directly edits the selected candidate workspace.
- `OpenEvolveAdapter`: the controller supplies OpenEvolve's prompt/history
  representation and accepts only exact SEARCH/REPLACE edits. The shared
  factorial controller—not OpenEvolve's native database or retention policy—
  controls parent selection, visibility, retention, and budgets.

### `runner.py`

Runs one locked proposal/evaluation opportunity:

1. checks the campaign's frozen scientific-runtime hash;
2. loads the crash-safe controller state;
3. selects/materializes the parent and visible candidates;
4. renders and saves the prompt and manifest;
5. invokes the selected framework adapter;
6. validates protected files and candidate edits;
7. snapshots the candidate by content hash;
8. evaluates valid candidates at most once;
9. applies retention, writes the event record, advances state, and saves the
   result artifact.

`recover_active_opportunity()` consumes an interrupted opportunity as a
  predeclared infrastructure failure. It does not delete, retry, or erase the
  original artifacts.

### `orchestration.py`

Implements campaign-level scheduling:

- protocol 1.0: serial blocked round-robin;
- protocol 1.1: C0–C3 in one block launch concurrently behind a start barrier,
  then N0 runs serially;
- one campaign lock prevents multiple orchestrators from writing the same
  campaign;
- partial waves resume only lagging conditions and record `recovery_subset`;
- every wave is append-logged in `parallel-rounds.jsonl`.

Do not launch four independent CLI processes. The orchestrator owns the
campaign lock and creates the internal C0–C3 worker pool.

### `campaign.py`, `evaluator.py`, `task_evaluators.py`, and `artifacts.py`

These modules provide:

- executed-on-target-backend calibration;
- immutable campaign construction from frozen protocol/task/framework inputs;
- copied task-support trees and content-addressed candidate snapshots;
- local evaluator command execution with controlled environment variables;
- metric extraction and qualification;
- evaluator-time and failure accounting;
- protected-file and runtime-hash checks.

### `validation.py`, `postsearch.py`, and `analysis.py`

`validate_campaign` is a fail-closed launch audit. It checks assignments,
paired seeds, candidate/support-tree identity, hashes, prompt treatment
skeletons, and untouched ready state.

After all runs finish:

- `export-layer-b` creates blinded, opaque mechanism-review packets;
- human reviewers produce annotations and an adjudicated annotation file;
- `score-layer-b` computes qualified mechanism outcomes and factorial estimates;
- `run-layer-c` performs the sealed final evaluation;
- `analysis.py` computes cell means, main effects, interaction, and descriptive
  outcomes.

### `modal_app.py` and `MODAL.md`

Modal execution is a separate serialized GPU path. It prepares or executes
calibration and candidate evaluation through a persistent Modal volume and
invokes the same controller against a remote campaign path. Protocol 1.1's
parallel local orchestration is intentionally not treated as a four-GPU Modal
launch; Modal's documented path remains serialized for controlled GPU use.

## 5. Frozen configurations

| Configuration | Purpose | Execution | Budget/checkpoints |
|---|---|---|---|
| `configs/protocols/dev.toml` | engineering smoke test | serial | 1 block, 4 opportunities |
| `configs/protocols/paper_v1.toml` | confirmatory paper stratum | serial | 3 blocks, 100 opportunities, checkpoints 20/40/60/80 |
| `configs/protocols/workshop_pilot_parallel_v1.toml` | local workshop pilot | C0–C3 parallel, N0 serial | 3 blocks, 30 opportunities, checkpoints 10/20 |

The workshop pilot uses GPT-5.6 Sol with xhigh reasoning and 30,000,000 total
Codex-token budget per run. Protocols are separate scientific strata. Do not
pool dev, workshop, and paper results as though they were identical protocols.

Framework configs are:

- `configs/frameworks/autoresearch.toml` for direct-edit Karpathy-style
  Autoresearch;
- `configs/frameworks/openevolve.toml` for the Greedy OpenEvolve adapter.

Tasks are:

- `configs/tasks/adderboard.toml` for the local AdderBoard parameter-minimizing
  task;
- `configs/tasks/karpathy_nanogpt.toml` for the pinned nanoGPT task and its
  hardware-specific objective.

## 6. One complete parallel wave

The parallel protocol finds the campaign-wide least-advanced opportunity count.
For the earliest eligible block it:

1. starts that block's C0, C1, C2, and C3 opportunities concurrently;
2. waits for all four factorial calls to finish or fail;
3. records the exact participant set in `parallel-rounds.jsonl`;
4. starts that block's N0 opportunity serially;
5. waits for N0 before selecting the next wave.

This is why a run may have completed opportunity 7 while another block is still
at opportunity 6: the scheduler always advances the least-advanced eligible
wave first.

## 7. Artifact map

Each campaign contains:

```text
campaign.json                 frozen identifiers and hashes
inputs/                       copied protocol/task/framework JSON
schedule.json                 blocked assignments and run IDs
validation.json               launch-audit result
parallel-rounds.jsonl         parallel-wave provenance
runs/<run-id>/manifest.json   assignment and runtime hash
runs/<run-id>/state.json      mutable crash-safe controller state
runs/<run-id>/events.jsonl    append-only scientific events
runs/<run-id>/candidates/     content-addressed candidate snapshots
runs/<run-id>/opportunities/  prompts, Codex JSONL, evaluations, results
```

The authoritative result is the durable `proposal_completed` event and its
`result.json`, not merely a Codex last-message file. Codex JSONL is used for
token recovery and transport diagnosis.

## 8. Standard commands

Run from the repository root:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli

$PY -m $CLI calibrate \
  --protocol experiments/c0c3_factorial/configs/protocols/dev.toml \
  --task experiments/c0c3_factorial/configs/tasks/adderboard.toml \
  --output data/c0c3/dev-adderboard-calibration \
  --python-bin "$PY"

$PY -m $CLI create \
  --protocol experiments/c0c3_factorial/configs/protocols/dev.toml \
  --task experiments/c0c3_factorial/configs/tasks/adderboard.toml \
  --framework experiments/c0c3_factorial/configs/frameworks/autoresearch.toml \
  --baseline data/c0c3/dev-adderboard-calibration/baseline.json \
  --output data/c0c3/dev-adderboard-campaign

$PY -m $CLI validate --campaign data/c0c3/dev-adderboard-campaign
$PY -m $CLI status --campaign data/c0c3/dev-adderboard-campaign
```

For protocol 1.0 use `run-next` or `run-campaign`. For protocol 1.1 use
`run-parallel-next` or `run-parallel-campaign`:

```bash
$PY -m $CLI run-parallel-campaign \
  --campaign data/c0c3/workshop-pilot-parallel-adderboard-autoresearch-campaign \
  --python-bin "$PY"
```

Before any launch, record the protocol/task/framework hashes, git revision,
Codex version, Python environment, operating system, and evaluator backend.

## 9. Recovery and operational rules

If a command exits nonzero, do not immediately retry. First check:

```bash
$PY -m $CLI status --campaign <campaign>
tail -n 20 <campaign>/parallel-rounds.jsonl
```

Inspect the host process, active `state.json`, opportunity directory, Codex
JSONL, stderr, and evaluator logs. Once it is certain no writer remains, use:

```bash
$PY -m $CLI recover-active \
  --campaign <campaign> \
  --run-id <exact-run-id> \
  --reason 'host process terminated after opportunity start'
```

For a partial parallel wave, recover every run whose state is active. The next
parallel command will run only lagging peers and mark the wave as a recovery
subset. Never delete artifacts, retry a charged opportunity, run two campaign
orchestrators, or manually edit `state.json`, `events.jsonl`, or
`parallel-rounds.jsonl`.

`status=running` means the run has not exhausted its budget; it does not prove
that a worker is currently executing. A healthy campaign has a runner process,
increasing proposal/evaluation counts, and each `parallel_wave_started` event
eventually followed by `parallel_wave_completed`.

## 10. What not to regress or pool blindly

- Do not replace the inherited `architecture_discovery` system with the C0–C3
  package; they serve different experiment layers and coexist in this
  repository.
- Do not treat old architecture-discovery trajectories as C0–C3 factorial data.
- Do not pool protocol 1.0, 1.1, or dev outcomes without an explicit stratum
  analysis.
- Do not expose sealed Layer B or Layer C artifacts to proposal agents or
  blinded reviewers before the required stage.
- Do not use the current local campaign outputs as source code or as a fresh
  calibration.

## 11. Recommended reading order for the next agent

1. This handoff.
2. `README.md`.
3. `PROTOCOL.md`.
4. `RUNBOOK.md`.
5. `FRAMEWORKS_AND_TASKS.md`.
6. `COMPLETION_AUDIT.md`.
7. `PAPER_NOTES.md`.
8. `spec.py`, `state.py`, `runner.py`, and `orchestration.py`.

The implementation and tests in this repository are the executable
authority. If a
legacy note or inherited architecture-discovery document conflicts with the
selected C0–C3 protocol, preserve the campaign's recorded protocol hash and
disclose the discrepancy rather than silently changing it.
