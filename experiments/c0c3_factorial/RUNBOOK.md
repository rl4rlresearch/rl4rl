# C0–C3 runbook

Run every command from the repository root. Paths below are repository-relative
or operator-chosen; no user-specific absolute path is embedded in a campaign.

## Goal-driven execution rule

When the operator assigns an end-to-end goal, carry it through to its stated
and verified end state without inserting an unrequested human checkpoint. In
particular, a goal that says to start, run, launch, resume, or continue a
campaign is itself authorization to perform that action after its required
validation and safety gates pass. Do not stop after preparation or validation
to ask the operator to confirm the launch again. This authorization persists
across automatic goal continuations and agent turns; assume the operator may be
away.

Use repository evidence and reasonable in-scope defaults to resolve ordinary
ambiguity, repair failed gates when possible, and keep working. Stop for input
only when the operator explicitly requested a hold or decision, essential
information or authority cannot be inferred, or an external permission
boundary makes the action impossible. A tool-level approval mechanism should
cite the original operator instruction directly and must not be preceded by a
duplicate chat confirmation request.

For unified v3, read [UNIFIED_V3.md](UNIFIED_V3.md). Use
`configs/protocols/unified_v3.toml` with either a v3 framework adapter. After
campaign creation, deliberately run `snapshot-v3-prompts`, then `validate` and
`v3-health`; no v3 start command is automatic. Prefix pairs are owned jointly
until their first intervention and become independently controllable after it.

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

For the continuous-session, one-block primary Autoresearch run, use:

```bash
PROTOCOL=$C0C3/configs/protocols/workshop_primary_block1_independent_continuous_v1.toml
TASK=$C0C3/configs/tasks/adderboard.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch_continuous.toml
OUT=data/c0c3/workshop-primary-block1-independent-continuous-adderboard-autoresearch
```

Protocol 1.4 pre-creates dormant extension assignments but its primary launcher
advances only Block 1 C0–C3. Do not add `--without-no-search` when creating this
campaign: pre-creating N0 under the frozen hashes is what makes a later N0 stage
comparable without making it part of the primary run.

For the separate subject-neutral protocol-1.5 Block 1 run, use:

```bash
PROTOCOL=$C0C3/configs/protocols/workshop_primary_block1_independent_continuous_v1_5.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_transformer.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch_continuous_v1_5.toml
OUT=data/c0c3/transformer-optimization-v1-5
```

This preset creates the same dormant extension structure as protocol 1.4, but
each scheduled run has its own controller and cooperative pause/resume
lifecycle. It must use a fresh calibration and campaign. Do not reuse
protocol-1.4 artifacts, and do not expose the internal campaign directory or
documentation to a subject session.

For prospective Greedy OpenEvolve protocol 2.0, use:

```bash
PROTOCOL=$C0C3/configs/protocols/controlled_openevolve_transformer_v2.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_mps.toml
FRAMEWORK=$C0C3/configs/frameworks/openevolve_v2.toml
OUT=data/c0c3/controlled-openevolve-transformer-v2-mps
```

Protocol 2.0 creates three C0–C3-only blocks and no N0. It uses bounded
ephemeral Codex proposals, the 1,644-parameter pair-token parent, and a strict
5,000-step-compatible learned-transformer evaluator. See
[OPENEVOLVE_V2.md](OPENEVOLVE_V2.md) before calibration or launch.

For artifact-clean continuous Autoresearch protocol 1.7, use:

```bash
PROTOCOL=$C0C3/configs/protocols/workshop_codex1644_source_only_v1_7.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_pair_transformer_codex1644_source_only.toml
FRAMEWORK=$C0C3/configs/frameworks/autoresearch_confined_v1_7.toml
OUT=data/c0c3/transformer-optimization-v1-7-source-only
```

For artifact-clean Greedy OpenEvolve protocol 2.1, use:

```bash
PROTOCOL=$C0C3/configs/protocols/controlled_openevolve_transformer_v2_1.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml
FRAMEWORK=$C0C3/configs/frameworks/openevolve_v2_1.toml
OUT=data/c0c3/controlled-openevolve-transformer-v2-1-mps
```

Both require a fresh calibration and campaign. The addition-task v1.7 preset
creates two blocks/eight runs; the v2.1 preset creates five blocks/twenty runs.
Neither creates N0 or copies the supplied trained checkpoint into a subject
workspace. Read [ARTIFACT_CLEAN_PROTOCOLS.md](ARTIFACT_CLEAN_PROTOCOLS.md)
before launch.

Verify prerequisites:

```bash
codex --version
$PY -c 'import torch; print(torch.__version__)'
$PY -m pytest -q tests/test_c0c3_factorial_core.py tests/test_c0c3_execution.py
```

Codex must already be authenticated. Scientific calls use the model and
reasoning effort frozen in the protocol TOML, not a user config default.
For the protocol-2.0 MPS task, also require
`torch.backends.mps.is_available()` to be true in this exact runtime; never
silently substitute CPU or Modal after calibration. Run this check from the
ordinary terminal that will launch the supervisor: restricted tool sandboxes
can hide Metal even when the same interpreter can use it outside the sandbox.

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

For AdderBoard, calibration verifies the immutable seed checkpoint supplied by
the task source. Candidate opportunities remain fresh-training evaluations and
never inherit that checkpoint. One calibration can be reused for both framework
campaigns only when protocol, task source, task configuration, and backend are
identical. A protocol change changes the protocol hash. Use a new calibration
when the amendment can affect calibration validity or backend comparability;
otherwise an operator-authorized in-place continuation may retain the original
calibration with the amendment boundary recorded.

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

By default protocols 1.0–1.6 create C0–C3 plus N0 in every block.
`--without-no-search` is available only for engineering diagnostics in those
protocols. Protocols 1.7, 2.0, and 2.1 freeze `include_no_search=false`, create
only C0–C3, and reject an attempted N0 override.

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

### Protocol 1.3 staged one-block primary

Run one primary Block 1 C0–C3 wave:

```bash
$PY -m $CLI run-staged-next \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage factorial
```

Run the complete 200-wave primary stage:

```bash
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage factorial
```

The runner launches only Block 1 C0–C3 concurrently. It never launches N0 or a
later block implicitly. The frozen primary must finish before the CLI permits
an extension. After it completes, the operator may explicitly run a prespecified
dormant stage:

```bash
# Optional descriptive N0 for Block 1
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage no-search

# Optional factorial replication for Block 2
$PY -m $CLI run-staged-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 2 --stage factorial
```

Use Block 3 identically. Record the decision timestamp and reason before the
first optional-stage call. If the choice used Block 1 outcomes, label later
blocks as adaptively collected extensions. Never rewrite the original Block 1
result as if three blocks had been the primary sample all along.

### Protocol 1.4 staged independently advancing primary

Protocol 1.4's primary launcher starts C0–C3 together once, then each trajectory
continues through its own opportunities as soon as its own evaluation finishes.
There is no per-opportunity round barrier and no N0 call:

```bash
$PY -m $CLI run-staged-independent-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage factorial
```

Keep this single command as the campaign writer. It internally owns four
distinct run directories and continuous Codex sessions; do not replace it with
four shell commands. A completed C0 trajectory can finish while another
condition is still running, which is expected. `status` shows per-run progress,
and `independent-trajectories.jsonl` records the initial group and the final
completion record for each trajectory.

After the primary stage, optional prespecified stages use the same launcher:

```bash
# Optional descriptive Block 1 N0
$PY -m $CLI run-staged-independent-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 1 --stage no-search

# Optional Block 2 factorial replication
$PY -m $CLI run-staged-independent-campaign \
  --campaign "$OUT-campaign" --python-bin "$PY" \
  --block 2 --stage factorial
```

Record the decision timestamp and reason before the first extension call. An
outcome-informed decision to activate it is an adaptive extension.

### Protocol 1.5 individually controlled trajectories

Protocol 1.5 does not have one campaign process that owns C0–C3. Start one
controller per scheduled run ID; each controller runs until its own budget is
complete or it receives a cooperative pause request. Use one terminal per run
to start all predeclared Block 1/2/3 C0–C3 run IDs under the same operational
plan; all such factorial trajectories may run concurrently.
Find the exact IDs with `status` or `schedule.json`, then run one command per
terminal:

```bash
$PY -m $CLI start-staged-trajectory \
  --campaign "$OUT-campaign" \
  --run-id '<exact-block-1-c0-run-id>' \
  --python-bin "$PY"
```

Repeat with the C1, C2, and C3 run IDs. These commands may run concurrently;
the lock is per run, not per campaign. A second start for the same run is
rejected. Do not use `run-staged-independent-campaign` or `run-one` for this
protocol.

To pause one active trajectory safely, leave its controller terminal running
and issue this from another terminal:

```bash
$PY -m $CLI pause-staged-trajectory \
  --campaign "$OUT-campaign" \
  --run-id '<exact-run-id>' \
  --reason 'operator-requested safe pause'
```

The request does not terminate an active Codex or evaluator call. It takes
effect before the next opportunity; the controller prints `"status": "paused"`
when the committed opportunity has finished. Do not use Ctrl-C as a pause. If
an interruption leaves `state.active` populated, recover that opportunity
instead.

Resume only that run with:

```bash
$PY -m $CLI resume-staged-trajectory \
  --campaign "$OUT-campaign" \
  --run-id '<exact-run-id>' \
  --python-bin "$PY"
```

The continuous Codex session, selected candidate, budgets, and next opportunity
remain attached to that run. Peers do not stop or restart. N0 remains locked
until the required factorial trajectories finish, while predeclared factorial
blocks may run concurrently. Record any adaptive extension decision before
its first start.

### Protocol 1.6 confined three-block trajectories

Use the same individual `start-staged-trajectory`,
`pause-staged-trajectory`, and `resume-staged-trajectory` commands as protocol
1.5. Create a fresh calibration and campaign with
`workshop_codex1644_confined_v1_6.toml`,
`ten_digit_addition_pair_transformer_codex1644_confined.toml`, and
`autoresearch_confined_v1_6.toml`. The campaign must contain three blocks and
the operational roster must select exactly C0–C3 from Blocks 1–3 (twelve jobs),
never N0. All twelve controllers may run simultaneously. The runtime admits at
most three evaluator processes from this campaign through crash-releasing file
locks. Each evaluator also acquires one of twelve host-wide slots shared by every
local protocol-1.6/1.7/2.0/2.1 campaign, so starting another campaign cannot
silently double Mac training concurrency. A queued trainer is healthy and does
not spend its evaluator timeout or evaluator-time budget while waiting.
At 500M reported tokens, each controller exits normally with
`token_threshold_reached`; the durable supervisor resumes that run. Its first
resumed prompt contains the one continuation notice, and subsequent prompts
contain no token-budget field or language. Do not manually retry or delete the
threshold-crossing opportunity.

For detached local operation use `experiments/c0c3_overnight.py` with
`RL4RL_OVERNIGHT_PROFILE=1644-confined`. Run `check` before `start`, then start
with `--recover-interrupted --all-running`. The supervisor uses `screen` plus
`caffeinate`, records a heartbeat and child PID for every trajectory, and
automatically charges/recoveries an interrupted opportunity before relaunching
only that trajectory. Inspect its `status`, `supervisor.log`, per-job logs, the
campaign thread registry, and each run's `state.json`/`events.jsonl`.

### Protocol 2.0 Greedy OpenEvolve trajectories

Use the same individual start/pause/resume commands as protocol 1.6. All twelve
C0–C3 run IDs are primary scope and may be supervised independently; no N0 run
exists. For detached local MPS operation, first create a detached worktree at
the exact committed launch revision, then use the `openevolve-v2` supervisor
profile:

```bash
git worktree add --detach /private/tmp/rl4rl-c0c3-openevolve-v2 HEAD

RL4RL_OVERNIGHT_PROFILE=openevolve-v2 \
  $PY experiments/c0c3_overnight.py check

RL4RL_OVERNIGHT_PROFILE=openevolve-v2 \
  $PY experiments/c0c3_overnight.py start \
  --recover-interrupted --all-running
```

Set `RL4RL_OPENEVOLVE_V2_CAMPAIGN` or `RL4RL_OPENEVOLVE_V2_RUNTIME` only before
the supervisor starts if nondefault paths are required. Use the same environment
variables for later status/pause/resume operations. The v2 profile does not
touch the primary or v1.6 profiles.

For evaluator-only Modal L4 offload, deploy `modal_hybrid_app.py`, select the
v2 Modal task TOML before calibration, and retain local Codex execution. Inspect
campaign-attributed GPU time with:

```bash
$PY -m $CLI modal-usage --campaign "$OUT-campaign"
architecture_discovery/.venv/bin/modal billing --help
```

MPS and Modal require distinct calibrations and campaigns. The local ledger is
not the authoritative account balance; use Modal Usage & Billing and set a hard
workspace or environment budget before launching.

### Protocols 1.7 and 2.1 artifact-clean trajectories

Use the same individual start/pause/resume and charged recovery commands as
protocols 1.6 and 2.0. All twelve C0–C3 run IDs are primary scope and no N0
exists. Create a detached runtime at the exact committed launch revision, then
use the corresponding durable supervisor profile:

```bash
# Autoresearch 1.7
git worktree add --detach /private/tmp/rl4rl-c0c3-autoresearch-v1-7 HEAD
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7 \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7 \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running

# Greedy OpenEvolve 2.1
git worktree add --detach /private/tmp/rl4rl-c0c3-openevolve-v2-1 HEAD
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1 \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1 \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
```

Override the prospective default campaign paths only before the supervisor
starts with `RL4RL_AUTORESEARCH_V17_CAMPAIGN` or
`RL4RL_OPENEVOLVE_V21_CAMPAIGN`. Use the matching `*_RUNTIME` variable when the
detached worktree is elsewhere. Run `check` before `start`; it verifies the
campaign inputs and local accelerator without changing trajectory state.

Before a trajectory's first start, the assumption-changing template remains
live in the main checkout. Save edits directly to the applicable v1.7 or v2.1
`assumption_changing.md`; no Git staging, commit, detached-worktree sync, or
accept-changes step is required. The first start snapshots the current text and
hash into that run. Subsequent edits apply only to trajectories that still have
zero proposals and have never started; started or resumed trajectories retain
their original snapshot. See `ARTIFACT_CLEAN_PROTOCOLS.md` for the exact files
and provenance behavior.

Both presets start Codex in Fast mode. Change subsequent Codex calls without
pausing or restarting the supervisor:

```bash
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7 $PY experiments/c0c3_overnight.py fast-mode off
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1 $PY experiments/c0c3_overnight.py fast-mode off
# Replace `off` with `on` or `status` as needed.
```

The selected tier is read immediately before every new or resumed Codex call
and is recorded in each proposal event. A call already in flight finishes on
the tier with which it began.

For the isolated nanoGPT campaigns, use the dedicated H100 deployment in
`MODAL.md`, then create separate detached runtimes for the four-block
Autoresearch v1.7 campaign and three-block Greedy OpenEvolve v2.1 campaign:

```bash
git worktree add --detach /private/tmp/rl4rl-c0c3-openevolve-v2-1-nanogpt HEAD
git -C /private/tmp/rl4rl-c0c3-openevolve-v2-1-nanogpt submodule update --init
git worktree add --detach /private/tmp/rl4rl-c0c3-autoresearch-v1-7-nanogpt HEAD
git -C /private/tmp/rl4rl-c0c3-autoresearch-v1-7-nanogpt submodule update --init
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7-nanogpt \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7-nanogpt \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-nanogpt \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-nanogpt \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
```

Both profiles support `fast-mode on|off|status`. The shared nanoGPT H100 app is
capped at three workers and does not take a local Mac evaluator slot.

For the separate fixed-exposure Fashion-MNIST strata, first follow
[FASHION_MNIST.md](FASHION_MNIST.md) to download and checksum the data, run two
protocol-specific MPS calibrations, create and validate the four-block v1.7 and
three-block v2.1 campaigns, and create their detached runtimes. Their durable
profiles are:

```bash
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7-fashion-mnist \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-fashion-mnist \
  $PY experiments/c0c3_overnight.py check
```

The corresponding `start --recover-interrupted --all-running` commands are
intentionally separate from preparation. Both use the shared local evaluator
pool and a stable dataset cache path; they do not invoke Modal.

### Operator-authorized Fashion-MNIST block expansion

An active C0–C3-only Fashion-MNIST campaign can be extended without restarting
or modifying its already-running trajectories. This is an in-place amendment:
the helper preserves the original controller protocol input, appends complete
seed-paired C0–C3 blocks, copies the byte-identical seed/task-support tree into
each new run, and records the boundary in `campaign-amendments.jsonl` plus an
`amendments/` snapshot directory. Do not edit `schedule.json` or run
directories manually.

For example, to extend the local Greedy OpenEvolve Fashion-MNIST campaign from three
to five blocks and launch only B4–B5 under a separate durable supervisor:

```bash
$PY experiments/c0c3_campaign_amend.py extend-blocks \
  --campaign data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign \
  --target-blocks 5 \
  --reason 'operator-authorized additional replication blocks'

RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-fashion-mnist-extension \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-fashion-mnist-extension \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
```

The extension supervisor controls B4–B5 only. The existing
`openevolve-v2.1-fashion-mnist` supervisor continues to own B1–B3, so the two
sets can be paused, resumed, or recovered independently while sharing the same
host-wide evaluator scheduler.

Inspect the shared local pool at any time without changing it:

```bash
$PY -m $CLI local-evaluator-status
```

The JSON reports the current operator-set host ceiling and the opportunity
holding each occupied slot. The shared scheduler defaults to twelve but has no
fixed maximum. `experiments/c0c3_overnight.py status` also prints the compact
occupied count. Protocols 1.7 and 2.1 set their campaign-local limit equal to their
declared block count; older campaigns retain their frozen local limit. Queue waiting happens before
the evaluator clock starts. Modal evaluator-only campaigns do not take local
host slots.

Unified v3 and semantic-v4 controllers also share a separate operator-sized
host pool for subject-agent calls. It defaults to 30 and has no fixed maximum.
Evaluators retain their independent task and host pools. Inspect the agent pool
without changing state:

```bash
$PY -m $CLI agent-worker-status
```

This is admission control, not a round scheduler: each trajectory takes the
next available slot and advances independently, with no wave or synchronization
barrier. A newly created campaign can join this pool without pausing any
campaign already using it. Only a controller running code from before the
shared pool existed needs one cooperative stop/start to adopt it; subsequent
campaign additions do not require a global drain.

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
For protocol 1.3, the same file also records `execution_stage`. After the
primary run, Block 1 C0–C3 should be completed while the other eleven runs
remain `ready` with zero proposals; this is expected, not a stalled campaign.
For protocol 1.4, inspect `independent-trajectories.jsonl` instead of
`parallel-rounds.jsonl`; the four primary run rows need not have matching
`proposals_used` while the launcher is active.
For protocols 1.5–1.7 and 2.0–2.1, inspect `trajectory-lifecycle.jsonl` plus each
run's `lifecycle.jsonl`; it records start, pause request, pause acknowledgement,
resume, transport stop, and completion for every independently controlled run.

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

For protocol 1.3, recover every active run, then invoke the same
`run-staged-next` or `run-staged-campaign` block/stage that was interrupted. The
runner selects only lagging peers within that stage and never advances a dormant
stage as part of recovery.
For protocol 1.4, recover every active run, then invoke the same
`run-staged-independent-campaign` block/stage. It starts only the unfinished
trajectories and each continues from its own next opportunity; it never creates
or retries a missing synchronized round.
For protocols 1.5–1.7 and 2.0–2.1, recover only the affected run, then invoke
`resume-staged-trajectory` with the same run ID. Do not recover or restart its
peers.

## 8. Export and adjudicate Layer B

For protocols 1.0–1.2, wait until `status` shows every run completed. For
protocols 1.3–1.5, first finish Block 1 C0–C3 and decide whether to activate any
optional extension. Activate every chosen extension before exporting. If none
is chosen, the dormant runs remain `ready` and the exporter seals only the four
frozen primary run IDs recorded in `campaign.json`:

```bash
$PY -m $CLI export-layer-b --campaign "$OUT-campaign"
```

This creates opaque, randomly ordered parent/candidate packets and a private
condition mapping. Do not show reviewers `sealed-layer-b/private/`, event logs,
run directories, or Layer A scores.
For protocols 1.3–1.5, `sealed-layer-b/scope.json` records the exact primary scope.
Once Layer B or C is created, the staged runner refuses to start an optional
extension; this prevents an unblinded primary review from driving additional
data collection.

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

Layer C is independent of Layer B adjudication but requires the campaign's
frozen analysis scope to be completed. Under protocols 1.3–1.5 that scope is the
four Block 1 primary runs; decide on extensions before creating Layer C:

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

## 12. Semantic-intervention v4 campaigns

Semantic v4 uses a separate launcher because its schedule contains arbitrary
intervention IDs rather than C0-C3 conditions. The complete copyable prepare,
validate, detached start, per-arm control, dashboard, and recovery commands are
in [SEMANTIC_INTERVENTIONS_V4.md](SEMANTIC_INTERVENTIONS_V4.md).

Important operational differences:

- one physical prefix leader supplies proposals 1-5 to every arm in a
  replicate;
- all post-fork arms are independently schedulable and controllable;
- a worker exception pauses only its arm, except that a prefix exception
  pauses that replicate until the shared state is recovered;
- campaign pause is cooperative and never deletes an active opportunity;
- shadow prefix usage is visible on conceptual trajectories but charged once
  in campaign physical totals;
- the periodic-refresh arm keeps its incumbent checkpoint but clears search
  population, visible history, and session state every five proposals;
- `semantic_intervention_overnight.py status` is the authoritative quick view.
