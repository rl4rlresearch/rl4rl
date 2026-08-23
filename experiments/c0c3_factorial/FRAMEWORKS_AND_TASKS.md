# Framework and task interfaces

The factorial controller deliberately separates **strategy/state** from
**proposal interface** and **task infrastructure**. That separation follows the
central lesson of FML-bench: comparisons become interpretable only when search
strategy is not entangled with evaluator and execution plumbing.

## 1. Framework-neutral controller

`SearchController` owns all experimental factors:

- visible single/portfolio state;
- selected parent;
- scheduled proposal type;
- proposal/evaluation/token/compute budgets;
- validity, retention, and stopping;
- logs and recovery.

A framework adapter receives a fully rendered controlled prompt, selected
parent workspace, visible read-only candidates, model contract, and log path. It
may produce exactly one candidate and metadata. It may not sample parents,
retain candidates, run evaluators, change budgets, or expose Layer B/C.

## 2. Karpathy-style direct Autoresearch adapter

Configuration: `configs/frameworks/autoresearch.toml`

The selected parent is a writable task workspace. Portfolio evidence appears in
`.factorial-visible/slot-N/` and is made read-only. Codex runs with
`workspace-write` and directly edits only task-declared editable files. The
controller hashes protected files before and after the call, snapshots editable
files, and rejects a no-op, duplicate, missing file, symlink, or protected edit.

This adapter captures Karpathy Autoresearch’s essential proposal interface:
an agent directly changes the research program and receives the task score. It
does **not** reuse the older long-lived `PROGRAM.md` runner, automation/microtrial
system, or conversational history; those would add uncontrolled strategy and
memory on top of C0–C3.

Edit mode: `direct_workspace`.

## 3. Controlled OpenEvolve adapter

Configuration: `configs/frameworks/openevolve.toml`

This adapter uses the vendored OpenEvolve implementation’s actual:

- `PromptConfig` and `PromptSampler`;
- population/history prompt representation;
- exact SEARCH/REPLACE extraction;
- diff application utilities.

All editable files are serialized into a deterministic multi-file bundle.
Visible portfolio candidates become OpenEvolve previous/top programs;
non-parent members become inspirations. Codex sees a read-only empty execution
workspace and must return exact SEARCH/REPLACE blocks. The adapter applies them
to the selected parent bundle and restores the declared file set.

Native OpenEvolve database sampling, island/population updates, and retention
are intentionally not used: those are precisely the memory/selection variables
controlled by C0–C3. Calling this “full native OpenEvolve” would be inaccurate;
the paper should call it the **controlled OpenEvolve proposal adapter**.

Edit mode: `search_replace_diff`.

### Protocol-2.0 controlled OpenEvolve adapter

Configuration: `configs/frameworks/openevolve_v2.toml`

The v2 adapter keeps OpenEvolve's prompt sampler and SEARCH/REPLACE interface,
but uses a subject-neutral transformer program, strict all-block patch
application, free-form mechanism provenance, and a bounded evidence ledger. It
shows each retained source once instead of copying the same programs into
previous/top/inspiration sections. Codex runs ephemerally in an opaque,
read-only, network-disabled workspace; only the resulting patch is applied to
the candidate workspace. See [OPENEVOLVE_V2.md](OPENEVOLVE_V2.md).

### Protocol-2.1 artifact-clean OpenEvolve adapter

Configuration: `configs/frameworks/openevolve_v2_1.toml`

The v2.1 adapter keeps the same OpenEvolve sampler and strict patch parser while
rendering source, public evidence, mechanism history, and response requirements
once each. It receives no controller budgets/horizon, selection counts, raw
runner fields, nonexistent reference paths, run seed, or trained checkpoint.
See [ARTIFACT_CLEAN_PROTOCOLS.md](ARTIFACT_CLEAN_PROTOCOLS.md).

## 4. Adding another research framework

Adding MAP-Elites, Go-Explore, Islands, curiosity, or another Heuresis-style
proposal strategy should require a narrow adapter rather than a controller fork:

1. Add a stable value to `FrameworkKind`.
2. Add a TOML with `framework_id`, versioned `adapter`, `prompt_profile`, and
   explicit `edit_mode`.
3. Implement one adapter with the existing `propose(...) -> ProposalExecution`
   boundary.
4. Dispatch it in `make_framework_adapter`.
5. Add a fake-Codex end-to-end test proving one candidate, protected-file
   containment, metadata parsing, and usage accounting.
6. Decide whether the adapter changes a factorial treatment. If it owns memory,
   parent selection, retention, or evaluator feedback, disable those native
   parts or define a different experiment—do not silently compose them.

The task, prompt renderer, controller, campaign, analysis, and Modal code should
not change for a normal new adapter.

## 5. Task contract

A task TOML defines:

- portable ID and display name;
- seed source (repository-relative or `${ENVIRONMENT_VARIABLE}`);
- one or more safe relative editable paths;
- Layer A evaluator command and metric parser;
- objective metric and maximize/minimize direction;
- optional qualification metric and threshold;
- exact public feedback fields;
- Layer C command;
- preferred backend.

Campaign creation copies the full support source, snapshots only editable files,
and hashes both. Every evaluation starts from a fresh support copy plus one
candidate snapshot; task checkpoints are removed to prevent cross-candidate
contamination. The evaluator owns no search state and emits at most one result.

For minimization, the controller stores `fitness = -objective`; logs retain the
human-readable positive metric. A candidate is valid only if the evaluator exits
zero, emits the objective, and meets any qualification threshold.

## 6. AdderBoard task

Configuration: `configs/tasks/adderboard.toml`

Research object: compress a trainable autoregressive transformer for 10-digit
addition while preserving correctness.

Editable files:

- `src/model.py`
- `src/data.py`
- `src/train.py`
- `submission.py`

Layer A:

1. Run candidate training from a clean workspace.
2. Run the trusted AdderBoard verifier on 10,000 cases with seed `2025`.
3. Require accuracy `>= 0.99`.
4. Among qualified candidates, minimize unique parameters.
5. Expose accuracy, parameters, and training steps.

Layer C reruns the trusted verifier on 10,000 cases with disjoint seed
`8,724,319`. Candidates cannot edit the verifier; its repository tree is part of
the scientific-runtime hash.

AdderBoard is useful because evaluator calls are cheap enough for dense search,
the objective is exact and interpretable, and architectural compression can
produce multiple mechanisms. It is also narrow and synthetic, so conclusions
must be checked on a qualitatively different task.

### Protocol-1.5 subject-neutral variant

Configuration: `configs/tasks/ten_digit_addition_transformer.toml`

This variant uses the same fixed 10-digit-addition verifier but deliberately
does not expose the benchmark name to the research agent. Its task-support tree
contains only the three editable source files, the immutable seed checkpoint,
and a protected generic decoder. Historical README, report, plots, result, and
handoff artifacts are excluded so the subject is not handed prior solutions or
parameter targets.

`submission.py` is protected. The task evaluator requires positive learned
parameters, positive training provenance, a learned self-attention module that
participates in the forward pass, and no recognized direct digit/carry
transducer in model source. This changes the admissible model class relative to
`adderboard.toml`, so the two tasks must be analyzed as distinct strata.

Protocol 1.5 also freezes independently controlled trajectory execution and
lifecycle provenance; those controls do not alter the task or model-validity
standard described here.

### Protocol-2.0 pair-token variants

Configurations:

- `configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_mps.toml`
- `configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_modal.toml`

Both use the 1,644-parameter pair-token parent, protect `src/data.py`, and expose
only `src/model.py` and `src/train.py`. They add deterministic source preflight,
strict fresh best/last checkpoint provenance, source immutability, learned
attention execution/ablation, and a disjoint final seed. The MPS and Modal L4
variants are hardware-specific strata and require separate calibration and
campaigns; their results are not interchangeable replications.

### Protocol-1.7/2.1 source-only pair-token variants

Configurations:

- `configs/tasks/ten_digit_addition_pair_transformer_codex1644_source_only.toml`
- `configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml`
- `configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_modal.toml`

These use task adapter `ten_digit_addition_pair_transformer_v3`. It copies the
protected source and decoder wrapper but never copies the seed's
`checkpoints/best.pt`. Baseline calibration and every candidate evaluation
therefore train in evaluator-owned workspaces from fresh initialization.

## 7. Official Karpathy Autoresearch nanoGPT task

Configuration: `configs/tasks/karpathy_nanogpt.toml`

Research object: improve fixed-time single-H100 language-model pretraining on
the official Karpathy Autoresearch workload. Only `train.py` is editable. The
objective is lower validation bits per byte (`val_bpb`); public process metrics
include time, VRAM, MFU, tokens, steps, parameters, and depth.

This is a strong complement to AdderBoard because it changes:

- domain from exact algorithmic reasoning to language modeling;
- objective from parameter compression under a quality constraint to
  time-bounded validation quality;
- compute regime from a small local model to a modern GPU training workload;
- mechanism space to architectures, optimizers, schedules, batches, attention,
  and systems efficiency.

### Pin and prepare the task source

Choose one official commit before any paper calibration:

```bash
git clone https://github.com/karpathy/autoresearch.git ../autoresearch-pinned
git -C ../autoresearch-pinned checkout '<full-commit-sha>'
export AUTORESEARCH_ROOT="$(cd ../autoresearch-pinned && pwd)"
git -C "$AUTORESEARCH_ROOT" rev-parse HEAD
```

Do not use a moving `main` checkout. Campaign creation copies the complete
source and hashes it, but the paper should still report the upstream commit.

Official dependencies currently require a CUDA-capable NVIDIA GPU and include
PyTorch 2.9.1/CUDA 12.8 plus `kernels`, `rustbpe`, and data tooling. The Modal
image pins these dependencies. `prepare.py` downloads training shards and a
pinned validation shard and trains the tokenizer into the persistent
`~/.cache/autoresearch` Volume. See [MODAL.md](MODAL.md).

The fixed five-minute time budget makes results hardware-specific. Calibrate and
execute every cell in a stratum on the same GPU type and software image. Do not
combine H100 and local/fork results.

### Layer C limitation

The official task exposes one pinned validation procedure, and the current
Layer C command reruns it from scratch. This measures replication stability, not
unseen-distribution generalization. Report it as such. Creating a new hidden
shard would change the upstream task and must be preregistered as a future task
version rather than improvised after seeing results.

## 8. Adding another ML task

Prefer tasks with:

- one trusted scalar objective and deterministic qualification rule;
- bounded evaluator wall time;
- a small, reviewable editable surface;
- enough mechanism diversity for Layer B clustering;
- a fixed public split and genuinely sealed final split where possible;
- licenses that allow source/data archival;
- runtime feasible for at least three complete blocks.

Implementation steps:

1. Add a task TOML; do not add condition-specific fields.
2. If needed, add a trusted wrapper in `task_evaluators.py` that emits
   `{"metrics": ...}` JSON.
3. Ensure all fixed evaluator files are protected and included in runtime/task
   hashes.
4. Add a toy/fixture integration test for parsing, qualification, objective
   sign, timeout, and fresh-workspace behavior.
5. Calibrate on the target backend.
6. Complete the dev protocol for both framework adapters before paper runs.
7. Add a task-specific Layer B mechanism rubric to `PAPER_NOTES.md` before
   reviewers see packets.

FML-bench’s 18-task breadth is a useful long-term direction, but adding many
tasks with too few complete trajectories is weaker than two well-controlled,
fully archived strata under the current deadline.
