# Artifact-clean Autoresearch 1.7 and Greedy OpenEvolve 2.1

Protocols 1.7 and 2.1 are prospective, source-only successors to protocols 1.6
and 2.0. They preserve the C0–C3 search-state and proposal-policy factors while
removing controller information that is unnecessary for solving the transformer
optimization task. Existing campaigns retain their original runtime hashes and
behavior.

## Subject-visible contract

Neither protocol exposes:

- proposal number, total horizon, remaining proposals, remaining evaluations,
  tokens, token limits, evaluator seconds, or continuation notices;
- prompt-region comments, empty policy regions, fixed-capacity empty design
  placeholders, selection counts, internal candidate IDs, or controller paths;
- evaluator fields outside `accuracy`, `parameters`, and `training_steps`;
- raw provider return codes, adapter/preflight dictionaries, protected-file
  flags, or infrastructure failure details;
- block/run seeds, a workspace identity file, benchmark/study/condition labels,
  or a supplied trained checkpoint.

Only designs that actually exist are described. Autoresearch reference paths
are shown only when those paths exist in its workspace. OpenEvolve source is
embedded once in its patch prompt and therefore receives no filesystem source
path. Treatment-region hashes and redacted prompt skeletons are stored in the
controller-side prompt manifest rather than as subject-visible HTML comments.

## Autoresearch 1.7

V1.7 uses one continuous Codex conversation per trajectory. Its first message
contains the complete task contract. Later messages contain only the newly
available verification result, the current verified design state, and any
scheduled assumption-changing direction. This avoids replaying a twelve-result
window and the complete task contract into a conversation that already retains
them.

Autoresearch summaries remain free-form. The adapter extracts labeled Markdown
when available and otherwise retains useful response text, so absent exact
`HYPOTHESIS:`/`INTENDED_EDIT:` lines no longer become repeated missing-value
placeholders.

The controller creates an ordinary local Git baseline in the opaque subject
workspace and directs temporary/cache files inside that workspace. It does not
write `.workspace-identity` or expose any run-derived environment seed. The
source-only task-support tree contains `src/` and the protected decoder wrapper,
not `checkpoints/best.pt`. Calibration and candidate verification train in
separate evaluator workspaces.

The `max_total_tokens` TOML field is retained for controller-side usage
accounting compatibility. V1.7 neither exposes nor enforces it; the trajectory
ends through the 200-proposal, 200-verification, or evaluator-time limits.

## Greedy OpenEvolve 2.1

V2.1 retains a fresh ephemeral Codex call for every proposal and therefore
supplies a bounded recent-result window. Each result contains only public
metrics and a subject-level outcome explanation. Mechanism information appears
once in this result history; the duplicate mechanism ledger is removed.

The OpenEvolve prompt sampler supplies current and reference source once.
Metrics and hypotheses appear once in the common design summary. The response
metadata and SEARCH/REPLACE contract appear once in the system prompt and are
not appended again by the adapter. The strict exact-once patch parser and source
preflight remain unchanged.

V2.1 retains v2.0's internal 100M-token safety ceiling, 200 proposal/evaluation
limits, evaluator-time limit, and optional evaluator-only Modal transport.
V1.7 and v2.1 set campaign-local evaluator capacity equal to the campaign's
predeclared block count and join the twelve-slot host scheduler shared with other
active local campaigns.
None of those resource values is shown to the subject.

## Assumption-prompt editing boundary

The v1.7 and v2.1 assumption-changing templates are operator-editable until an
individual trajectory first starts. The ordinary files in the main checkout
are the live source:

- `templates/transformer_optimizer_v1_7/assumption_changing.md`;
- `templates/transformer_optimizer_openevolve_v2_1/assumption_changing.md`;
- `templates/nanogpt_optimizer_v1_7/assumption_changing.md`;
- `templates/nanogpt_optimizer_openevolve_v2_1/assumption_changing.md`.

Saving either file is sufficient; staging or committing it is not required for
an unstarted trajectory to receive the new text. At `trajectory_started`, the
controller copies the applicable file into that run's `subject-prompt/`
directory and records its SHA-256 in the lifecycle event. Every later
opportunity in that trajectory uses this private snapshot, so an operator edit
cannot silently change a trajectory already in progress. The two live files
are excluded from the artifact-clean scientific-runtime hash because their
per-trajectory snapshot hashes are the authoritative prompt provenance; all
other controller and prompt files remain runtime-hashed normally.

The durable v1.7/v2.1 supervisor passes the main checkout's template directory
to detached runtimes automatically. Direct starts also discover it from a
campaign stored beneath the main repository; an explicit
`RL4RL_C0C3_OPERATOR_PROMPT_ROOT` is available only for campaigns stored
elsewhere.

## Frozen presets

Autoresearch:

```bash
PROTOCOL=experiments/c0c3_factorial/configs/protocols/workshop_codex1644_source_only_v1_7.toml
TASK=experiments/c0c3_factorial/configs/tasks/ten_digit_addition_pair_transformer_codex1644_source_only.toml
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/autoresearch_confined_v1_7.toml
```

Local-MPS Greedy OpenEvolve:

```bash
PROTOCOL=experiments/c0c3_factorial/configs/protocols/controlled_openevolve_transformer_v2_1.toml
TASK=experiments/c0c3_factorial/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/openevolve_v2_1.toml
```

For evaluator-only Modal L4, replace the v2.1 MPS task with
`ten_digit_addition_pair_transformer_openevolve_v2_1_modal.toml` and perform a
separate target-backend calibration.

The addition-task v1.7 preset freezes two C0–C3-only blocks and v2.1 freezes
five; neither has N0. Both start in Codex Fast mode, with a controller-side
switch that takes effect on the next Codex call. Use the ordinary calibration,
creation, validation, and individual trajectory start/pause/resume commands in
`RUNBOOK.md`.

## Source-only nanoGPT task

Protocols 1.7 and 2.1 also support the pinned official Karpathy Autoresearch
source as a separate H100 task stratum. The subject receives only `train.py`
and the protected `prepare.py` utilities; upstream `program.md`, README, Git
history, and prior results are excluded. The objective is lower `val_bpb` after
the official five-minute measured training window. Addition and nanoGPT never
share task support, calibration, candidate state, or campaign directories.

The matching files are:

```text
configs/protocols/nanogpt_autoresearch_v1_7.toml
configs/protocols/nanogpt_openevolve_v2_1.toml
configs/tasks/karpathy_nanogpt_source_only_h100.toml
configs/frameworks/autoresearch_nanogpt_v1_7.toml
configs/frameworks/openevolve_nanogpt_v2_1.toml
```

Codex and campaign state stay local. Training is sent to the dedicated
`rl4rl-c0c3-nanogpt-evaluator-v1` Modal service, which permits at most three
H100 workers. Its leases use only the nanoGPT campaign pool and do not consume
the Mac's twelve local evaluator slots. Every call is recorded in
`modal-usage.jsonl`; Modal Usage & Billing remains authoritative for credits and
spend.

## Source-only Fashion-MNIST task

Protocols 1.7 and 2.1 also support a separate local-MPS image-classification
stratum. Subjects receive only `train.py`; the protected evaluator owns the
checksum-verified data, frozen 50k/10k train/validation split, exactly 100,000
presented training examples, fixed normalization, scoring, and the sealed
official test set. The objective ranks exact validation correct count first and
lower cross-entropy only on ties.

The matching files are:

```text
configs/protocols/fashion_mnist_autoresearch_v1_7.toml
configs/protocols/fashion_mnist_openevolve_v2_1.toml
configs/tasks/fashion_mnist_source_only_mps.toml
configs/frameworks/autoresearch_fashion_mnist_v1_7.toml
configs/frameworks/openevolve_fashion_mnist_v2_1.toml
```

The v1.7 preset has four C0–C3 blocks and the v2.1 preset has three, matching
the nanoGPT block geometry. Both are prepared but require an explicit dataset
download, target-Mac calibration, campaign creation, validation, and launch.
See [FASHION_MNIST.md](FASHION_MNIST.md).
