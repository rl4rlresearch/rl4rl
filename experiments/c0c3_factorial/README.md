# Controlled C0–C3 autonomous-research experiment

This package implements a blocked 2×2 factorial experiment over two properties
of an autonomous ML-research loop:

| | Ordinary proposal policy | Scheduled assumption-changing policy |
|---|---|---|
| Single incumbent | C0 | C1 |
| Portfolio memory (`K=4`) | C2 | C3 |

The scientific question is whether access to multiple live research branches,
scheduled pressure to change architectural assumptions, or their interaction
increases the number of distinct, valid mechanism families an agent discovers.
The primary outcome is **distinct Layer-B-qualified mechanism clusters per
run**. Protocols 1.0–1.6 can include an independent-proposal no-search baseline,
N0, outside the 2×2 contrasts. Protocols 1.7, 2.0, and 2.1 contain only C0–C3.

## Read in this order

1. [PROTOCOL.md](PROTOCOL.md) — frozen treatments, state transitions, controls,
   outcomes, and analysis.
2. [RUNBOOK.md](RUNBOOK.md) — exact local commands from calibration through
   sealed analysis and recovery.
3. [FRAMEWORKS_AND_TASKS.md](FRAMEWORKS_AND_TASKS.md) — what “Karpathy
   Autoresearch” and “OpenEvolve” mean here, AdderBoard and nanoGPT setup, and
   extension interfaces.
4. [MODAL.md](MODAL.md) — target-backend calibration, persistent volumes, and
   serialized GPU execution for protocol 1.0. Parallel protocols 1.1, 1.3,
   1.4, 1.5, and 1.6 are currently local-only.
5. [PAPER_NOTES.md](PAPER_NOTES.md) — literature, hypotheses, reviewer rubric,
   figures, statistical cautions, and paper-writing checklist.
6. [OPENEVOLVE_V2.md](OPENEVOLVE_V2.md) — the no-N0, validity-constrained
   controlled OpenEvolve protocols and launch commands.
7. [ARTIFACT_CLEAN_PROTOCOLS.md](ARTIFACT_CLEAN_PROTOCOLS.md) — the v1.7 and
   v2.1 source-only subject boundary and prompt-cleanliness guarantees.
8. [UNIFIED_V3.md](UNIFIED_V3.md) — the unified extensible successor,
   campaign-wide prompt snapshot, bounded capsules, and paired-prefix fork.

Future coding agents must also follow [AGENTS.md](AGENTS.md).

## Implementation map

- `spec.py`: strict TOML contracts, C0–C3 mapping, frozen rule identifiers, and
  deterministic blocked assignments.
- `state.py`: the framework-neutral search controller, budgets, selection,
  retention, event log, and crash-safe state.
- `prompts.py` plus `templates/`: one common prompt with exactly two marked
  treatment regions.
- `frameworks.py`: direct-edit Autoresearch and controlled OpenEvolve
  SEARCH/REPLACE adapters.
- `campaign.py`: portable calibration and immutable campaign construction.
- `runner.py`: one locked proposal/evaluation opportunity.
- `evaluator.py`: campaign-local evaluator limits plus one crash-releasing,
  twelve-slot host scheduler shared by every local 1.6/1.7/2.0/2.1 campaign.
- `orchestration.py`: versioned serial, synchronized-wave, and independently
  advancing parallel execution, plus campaign writer locks and append-only
  execution provenance.
- `validation.py`: fail-closed launch audit.
- `postsearch.py` and `analysis.py`: sealed Layer B review, Layer C evaluation,
  and factorial contrasts.
- `modal_app.py`: the same runner on a serialized Modal H100 worker.
- `configs/`: switchable protocols, tasks, and framework adapters.
- `v3.py` and `v3_analysis.py`: unified flexible controls, campaign-wide prompt
  snapshot, literal prefix sharing/forking, provenance, health, and audit.

## Scope presets

- `configs/protocols/dev.toml` is a four-opportunity, one-block engineering
  smoke test. It is not evidence for a paper.
- `configs/protocols/paper_v1.toml` is the frozen confirmatory configuration:
  three blocks, 100 opportunities per run, `K=4`, and transition checkpoints
  20/40/60/80. One task/framework campaign contains 15 runs including N0.
- `configs/protocols/workshop_pilot_parallel_v1.toml` is a separately versioned,
  deadline-bounded protocol: three blocks, 30 opportunities per run, `K=4`,
  transition checkpoints 10/20, GPT-5.6 Sol xhigh, and synchronized local C0–C3
  block rounds. It is potentially usable evidence under its own protocol label;
  it must never be pooled with `paper_v1` as if the execution rules and budgets
  were identical.
- `configs/protocols/workshop_pilot_parallel_continuous_v1.toml` is protocol
  1.2 for continuous-session Autoresearch: 200 opportunities per trajectory,
  an intervention every tenth opportunity, and one resumed Codex conversation
  per run. It is a separate stratum; see `CONTINUOUS_AUTORESEARCH.md` and do not
  pool it with protocol 1.1.
- `configs/protocols/workshop_primary_block1_continuous_v1.toml` is protocol
  1.3 and preserves the original synchronized-wave primary preset. Its primary
  stage runs only Block 1 C0–C3: four concurrent trajectories and 800 total
  opportunities. It creates but does not advance N0 or Blocks 2–3, preserving
  exact optional extensions under the same campaign hashes. Use only the staged
  commands documented in `CONTINUOUS_AUTORESEARCH.md`.
- `configs/protocols/workshop_primary_block1_independent_continuous_v1.toml`
  is protocol 1.4 and is the current launch preset. It keeps the same
  treatments, continuous conversations, budgets, seeds, primary scope, and
  dormant extensions as protocol 1.3, but C0–C3 start together and then each
  advances independently through all 200 opportunities rather than waiting at
  every shared wave boundary. It is a separate execution stratum and cannot be
  pooled with protocol 1.3 as if their timing rules were the same.
- `configs/protocols/workshop_primary_block1_independent_continuous_v1_5.toml`
  is protocol 1.5. It retains the protocol-1.4 execution geometry and factorial
  mapping but introduces a subject-neutral engineering prompt, a sanitized
  workspace, condition-private recent-result summaries, a protected generic
  decoder, learned-transformer contract validation, and one independently
  controlled process per scheduled trajectory. It is scientifically separate
  from 1.4 because the admissible model class, online evidence, and execution
  ownership changed. Its primary stage remains Block 1 C0–C3; N0 and Blocks 2–3
  are pre-created extensions; predeclared factorial blocks may be launched
  concurrently, while N0 remains dormant unless explicitly activated.
- `configs/protocols/workshop_codex1644_confined_v1_6.toml` is the replacement
  three-block 1,644-parent campaign. It preserves the v1.5 C0–C3 treatments and
  200-cycle continuous trajectories, but binds resumed processes to their real
  opaque cwd, disables user configuration/rules/network and extra temporary
  writable roots, registers one unique Codex thread per run, freezes inference
  preprocessing, retrains from an empty checkpoint directory, and allows at
  most three local evaluators from that campaign at once. All local campaigns
  also share one twelve-slot host scheduler. All twelve C0–C3 trajectories may
  remain active concurrently; evaluator queueing is condition-common
  machine-load control rather than a trajectory barrier. Its 500M-token value is a common
  subject-visible phase threshold: the controller returns at the crossing,
  resumes once with a minimal continuation notice, and omits token-budget
  language from every later prompt.
- `configs/protocols/controlled_openevolve_transformer_v2.toml` is the
  prospective controlled OpenEvolve replacement: three C0–C3-only blocks, 200
  bounded ephemeral proposals per run, the 1,644-parameter/5,000-step parent,
  neutral prompts, strict patch/source preflight, trained-attention checks,
  three campaign-local evaluator slots, the shared twelve-slot host scheduler,
  independent supervision, and optional evaluator-only Modal L4 offload. It is
  a new stratum and cannot be pooled with protocol 1.1.
- `configs/protocols/workshop_codex1644_source_only_v1_7.toml` is the
  artifact-clean continuous Autoresearch successor. Its current addition-task
  preset has two blocks/eight primary trajectories, no N0, no subject-visible
  resource/horizon accounting, source-only workspaces, and no token-based stop.
- `configs/protocols/controlled_openevolve_transformer_v2_1.toml` is the
  artifact-clean ephemeral OpenEvolve successor. It preserves v2.0's search
  geometry and evaluator controls while removing subject-visible orchestration
  artifacts and redundant prompt composition; its current addition-task preset
  has five blocks/twenty trajectories. In both 1.7 and 2.1, the main
  checkout's assumption-changing template remains live until each trajectory's
  first start and is then hashed and frozen privately for that trajectory.
- `configs/protocols/nanogpt_autoresearch_v1_7.toml` and
  `configs/protocols/nanogpt_openevolve_v2_1.toml` apply the same artifact-clean
  C0–C3 factors to a pinned, source-only official nanoGPT task. Codex remains
  local; a dedicated three-worker H100 evaluator service owns fixed-time
  training and is isolated from addition campaigns and local Mac evaluator
  slots.
- `configs/protocols/fashion_mnist_autoresearch_v1_7.toml` and
  `configs/protocols/fashion_mnist_openevolve_v2_1.toml` mirror nanoGPT's
  four-block/three-block artifact-clean geometry for fixed-exposure local-MPS
  image classification. The evaluator owns a frozen 50k/10k split, exactly
  100,000 presented training examples, and the sealed official test set. See
  [FASHION_MNIST.md](FASHION_MNIST.md); setup does not download, calibrate, or
  launch this task automatically.
- `configs/protocols/unified_v3.toml` is the single successor protocol for
  Autoresearch, OpenEvolve, plugin frameworks, and plugin tasks. Its supplied
  preset uses eight 100-proposal blocks and no N0. C0/C1 and C2/C3 literally
  share their pre-intervention trajectories and fork from identical state at
  the first intervention. See [UNIFIED_V3.md](UNIFIED_V3.md). No official v3
  run is launched by setup or validation.
- `configs/protocols/semantic_interventions_v4.toml` is the exploratory
  multi-arm expansion built on the unified v3 controller. Twenty-one semantic
  research operations receive three matched replicates, share one literal
  five-proposal prefix per replicate, and then run in five-proposal sessions.
  It adds candidate-editable bounded training ladders, non-selective
  developmental evidence, independent arm controls, and physical resource
  de-duplication. See [SEMANTIC_INTERVENTIONS_V4.md](SEMANTIC_INTERVENTIONS_V4.md).
  The supplied first-launch stratum is Fashion-MNIST/OpenEvolve; AdderBoard,
  nanoGPT, and Autoresearch are prepared but are not started automatically.

Run the dev protocol end to end before spending on the paper protocol. Do not
reinterpret dev results as a pilot effect estimate: its transition density and
budget are intentionally unlike the paper protocol.

## Non-negotiable invariants

- Protocol, prompt, adapter, evaluator, task-support, and controller changes may
  continue in an existing campaign when explicitly authorized by the operator.
  Preserve old artifacts, record the exact amendment boundary and affected run
  IDs, update the campaign's executable metadata consistently, and validate the
  continuation path before restarting writers. A fresh campaign remains an
  available design choice, not an automatic requirement.
- Use `run-next`/`run-campaign` for protocol 1.0 and
  `run-parallel-next`/`run-parallel-campaign` for protocol 1.1. Each command
  rejects a campaign frozen to the other rule. `run-one` and `run --run-id`
  exist for diagnostics and can bypass randomized ordering.
- Use `run-staged-next`/`run-staged-campaign` for protocol 1.3. Use
  `run-staged-independent-campaign` for protocol 1.4. Use
  `start-staged-trajectory`, `pause-staged-trajectory`, and
  `resume-staged-trajectory` for protocols 1.5–1.7 and 2.0–2.1. Protocols 1.7,
  2.0, and 2.1 have no optional N0 stage and freeze every campaign-declared
  C0–C3 block as their analysis scope.
- Never expose Layer B annotations or Layer C results until every run in the
  campaign is completed.
- Do not delete, retry, or reuse a failed opportunity. Use `recover-active` for
  an interrupted active opportunity; the opportunity remains charged.
- Do not run the same campaign from local and Modal storage simultaneously.
- Keep one campaign orchestrator process for protocols 1.1–1.4. Protocols
  1.5–1.7 and 2.0–2.1 instead authorize one controller per scheduled run; a
  per-run lock rejects a second controller for the same trajectory, while peers
  may operate in parallel.
- Do not pool task or framework strata as if they were interchangeable
  replications. Report each task × framework stratum, then synthesize effects.

## Minimal local entry point

From the repository root:

```bash
PY=architecture_discovery/.venv/bin/python
C0C3=experiments/c0c3_factorial

$PY -m experiments.c0c3_factorial.cli --help
```

The full, copyable sequence is in [RUNBOOK.md](RUNBOOK.md).
