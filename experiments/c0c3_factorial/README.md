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
run**. An independent-proposal no-search baseline, N0, is measured separately
and is never included in the 2×2 contrasts.

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
- `orchestration.py`: versioned serial, synchronized-wave, and independently
  advancing parallel execution, plus campaign writer locks and append-only
  execution provenance.
- `validation.py`: fail-closed launch audit.
- `postsearch.py` and `analysis.py`: sealed Layer B review, Layer C evaluation,
  and factorial contrasts.
- `modal_app.py`: the same runner on a serialized Modal H100 worker.
- `configs/`: switchable protocols, tasks, and framework adapters.

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
  most three local evaluators at once. All twelve C0–C3 trajectories may remain
  active concurrently; the evaluator limit is condition-common machine-load
  control rather than a trajectory barrier.

Run the dev protocol end to end before spending on the paper protocol. Do not
reinterpret dev results as a pilot effect estimate: its transition density and
budget are intentionally unlike the paper protocol.

## Non-negotiable invariants

- Create and validate a new campaign after any protocol, prompt, adapter,
  evaluator, task-support, or controller change. Runtime hashes deliberately
  make old campaigns refuse the new code.
- Use `run-next`/`run-campaign` for protocol 1.0 and
  `run-parallel-next`/`run-parallel-campaign` for protocol 1.1. Each command
  rejects a campaign frozen to the other rule. `run-one` and `run --run-id`
  exist for diagnostics and can bypass randomized ordering.
- Use `run-staged-next`/`run-staged-campaign` for protocol 1.3. Use
  `run-staged-independent-campaign` for protocol 1.4. Use
  `start-staged-trajectory`, `pause-staged-trajectory`, and
  `resume-staged-trajectory` for protocols 1.5–1.6. Each primary Block 1 factorial
  stage must complete before an optional stage can start.
- Never expose Layer B annotations or Layer C results until every run in the
  campaign is completed.
- Do not delete, retry, or reuse a failed opportunity. Use `recover-active` for
  an interrupted active opportunity; the opportunity remains charged.
- Do not run the same campaign from local and Modal storage simultaneously.
- Keep one campaign orchestrator process for protocols 1.1–1.4. Protocols 1.5–1.6
  instead authorizes one controller per scheduled run; a per-run lock rejects a
  second controller for the same trajectory, while peers may operate in parallel.
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
