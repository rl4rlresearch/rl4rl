# Tiny AdderBoard v2.1

Operator-authorized September 5, 2026: five independent C0-C3 blocks, 200
proposal opportunities and 200 candidate evaluations per trajectory. No C4 or
N0. All twenty Sol/xhigh trajectories may generate concurrently. Fast service
tier is inherited from the ten-digit v2.1 protocol.

The search controller, fresh per-opportunity conversations, twelve-result
history, four-slot portfolio, strict selected-lineage improvement rule, and
every-tenth-opportunity C1/C3 intervention are the ten-digit v2.1 design.
There is no later-protocol paired prefix, semantic critic, or training ladder.
The assumption-changing prompt is copied byte-for-byte from ten-digit v2.1.

The shared task is exact addition of two integers in [0, 9999], minimizing
deduplicated learned parameters at >=99% exact accuracy. `train.py` exposes
model architecture, tokenization, optimizer, loss, clipping, batch size, and
`TRAINING_STEPS`. The seed uses 1,012 parameters and **400 optimizer steps**.
An independent 1,000-step cosine schedule horizon preserves its verified
trajectory; it does not execute extra updates. Both settings are editable.
Pair tokens are a starting representation, not a task restriction. The prompt
permits replacing them and coordinating arbitrary learned-transformer changes.

The protected evaluator owns hash-disjoint training/public/holdout sets, fresh
initialization, update counting, checkpoint writing/reloading, generic greedy
decoding, exact scoring, and attention-dependence checks. It uses 10,000 unique
public examples. A separate post-search command uses 10,000 disjoint holdout
examples; holdout data never enters search feedback. This four-digit data
generator is shared with the compact-seed verification, independently of the
later protocol's search or ladder rules. Target codecs receive only generated
answer tokens at inference, never operands or expected sums. No hardcoded
arithmetic solver is permitted in model or codec code.

## Execution and fallback

Modal app `rl4rl-tiny-adderboard-v21` runs CPU-only PyTorch 2.14.0 with **2 cores
and 4 GiB per evaluator**. Its function has no `max_containers` setting and
there is no client evaluator semaphore. Modal account/platform quotas still
apply. There is no shared subject-worker lease for this task.

Each evaluation first attempts Modal. Infrastructure failures fall back to
Windows, using a task-wide pool of exactly three process locks and two PyTorch
threads per local evaluator. Every new evaluation attempts Modal again;
evaluations queued for local capacity retry Modal every thirty seconds. A local
evaluation already running finishes normally. Invalid candidates and accuracy
failures are scientific results and never trigger a backend retry. Remote call
IDs and failures are recorded; known calls are cancelled before local fallback.
An unconfirmed cancellation stops that trajectory for recovery rather than
silently duplicating execution. Backend receipts remain outside subject prompts.

## Calibration evidence and limits

Initialization seed 20260828 is the previously verified compact-seed
initialization and is used for baseline calibration. The assignment seed was
changed from the ten-digit study's 20260824 to this value before campaign
creation. Block seeds remain independently derived by the unchanged v2.1 rule
and matched across the four conditions in each block.

Initial validation of this adapter: Windows 99.80% public accuracy, Modal
100%, both at 400 steps, 1,012 parameters, and zero accuracy after attention
ablation. Modal evaluation took 24.0 seconds (17.1 seconds training), with
29.2 seconds client wall time including dispatch/result transfer. These are
single-calibration timings, not guarantees for proposed architectures.

Final source calibration again reached 100% on Modal at 400 steps. It took
35.9 evaluator seconds, including 16.6 seconds training. Budget about thirty
seconds per seed evaluation, with dispatch and process-start variation.

Seed sensitivity is material: a separate Windows check at initialization
20260824 reached 26.29% at 400 steps. Baseline qualification does not imply
that every initialization qualifies. Per-proposal fresh-run accuracy is
measured normally; the LLM can change the training budget. CPU platforms may
produce different floating-point trajectories even with the same seed.

## Launch and inspection

Configs: `configs/protocols/tiny_adderboard_v2_1.toml`,
`configs/tasks/tiny_adderboard_v2_1_modal.toml`, and
`configs/frameworks/tiny_adderboard_v2_1.toml`.

`start_tiny_v21.ps1` validates the untouched campaign then launches twenty
hidden, independent Windows processes through the existing v2.1 trajectory
controller. Each continues until its budget is exhausted or a declared stop
condition occurs. `windows-launch.json` records PIDs; `workers/` contains logs.
Do not blindly rerun the launcher: it rejects a pre-existing launch receipt.
Use the existing pause/resume/recover CLI to operate individual trajectories.
The fallback works while the Windows host and these controllers remain running.

Scientific source and settings must be finalized before creating the campaign;
runtime hashes intentionally reject later in-place drift.

## Verification on the restored Windows checkout

Package Ruff checks, eight new task/lock regression tests, all eight rendered
condition/intervention prompt checks, real Sol/xhigh transport, local training,
and two remote CPU calibrations passed. With Windows extended-length temporary
paths, 52 of 53 core tests passed; the remaining test imports the legacy C4
tool's POSIX-only `fcntl`. The normal core/execution invocation also encounters
legacy long-path and executable-shebang fixtures. Repository-wide pytest was
run, including collectable tests; older overnight/autoresearch/lifecycle modules
still have POSIX-only imports and legacy suites have Windows fixture failures.
The Mac `architecture_discovery/.venv/bin/python` interpreter is absent here.
These limitations are recorded rather than calling the full suite green.
Raw verification logs and calibration bundles are under `outputs/tiny-v21/`.
# Windows recovery amendment (2026-09-05)

The operator authorized safe recovery and resumption of all twenty trajectories.
Windows state reads and atomic replacements now retry sharing/access errors for
up to five seconds, using the same saved payload rather than repeating a proposal.
Other errors still fail. Recovery snapshots, old/new runtime hashes, and exact
run boundaries are retained in the campaign's `recovery-20260905` directory and
`campaign-amendments.jsonl`. Existing Modal responses are retrieved using their
saved function-call IDs and checked against payload hashes and candidate identity;
completed evaluations enter the ordinary retention/accounting transition once.
Only a proposal without completed recoverable work uses charged interruption
recovery. The evaluator, training, prompts, conditions, and budgets are unchanged.
