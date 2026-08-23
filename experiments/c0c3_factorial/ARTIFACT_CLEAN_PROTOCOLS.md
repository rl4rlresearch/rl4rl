# Artifact-clean Autoresearch 1.7 and OpenEvolve 2.1

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

## OpenEvolve 2.1

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
limits, evaluator-time limit, three local evaluator slots, and optional
evaluator-only Modal transport. None of those resource values is shown to the
subject.

## Frozen presets

Autoresearch:

```bash
PROTOCOL=experiments/c0c3_factorial/configs/protocols/workshop_codex1644_source_only_v1_7.toml
TASK=experiments/c0c3_factorial/configs/tasks/ten_digit_addition_pair_transformer_codex1644_source_only.toml
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/autoresearch_confined_v1_7.toml
```

Local-MPS OpenEvolve:

```bash
PROTOCOL=experiments/c0c3_factorial/configs/protocols/controlled_openevolve_transformer_v2_1.toml
TASK=experiments/c0c3_factorial/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml
FRAMEWORK=experiments/c0c3_factorial/configs/frameworks/openevolve_v2_1.toml
```

For evaluator-only Modal L4, replace the v2.1 MPS task with
`ten_digit_addition_pair_transformer_openevolve_v2_1_modal.toml` and perform a
separate target-backend calibration.

Both protocols freeze three C0–C3-only blocks: twelve primary trajectories and
no N0 assignment. Use the ordinary calibration, creation, validation, and
individual trajectory start/pause/resume commands in `RUNBOOK.md`.
