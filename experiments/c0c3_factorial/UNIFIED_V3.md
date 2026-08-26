# Unified v3 protocol

Unified v3 is one protocol contract for direct-edit Autoresearch, controlled
OpenEvolve, future search architectures, and new tasks. Framework and task
adapters remain separate configuration inputs, but they run through the same
campaign, state, prompt, evaluation, accounting, review, and audit interfaces.

No official v3 campaign is created or started by this implementation.

## Paired-prefix intervention design

Each block contains two literal shared-prefix pairs:

- C0 is the physical C0/C1 trajectory before the first assumption-changing
  opportunity; C1 inherits every proposal, candidate, evaluation, prompt,
  state transition, and usage receipt from that trajectory.
- C2 is equivalently the physical C2/C3 trajectory before the first
  assumption-changing opportunity.
- At the first configured intervention, each pair has byte-identical candidate
  and controller history. C0 and C2 receive ordinary direction while C1 and C3
  receive assumption-changing direction, and all four then advance
  independently.

`paired-prefix.json` records the fork, and `paired-prefix-events.jsonl` records
every inheritance operation. Shared prefix costs appear in both trajectory
histories but carry `resource_accounting=shared_prefix_charge_once_to_pair`;
`summarize-v3` counts each physical prefix call only once.

The mirror is crash-resumable. Its opportunity directory is materialized
atomically, source events are content-addressed and deduplicated, and shadow
state is committed last. A process interruption therefore resumes the pending
inheritance instead of sampling another proposal.

## Prompt and context behavior

The only new immutable launch artifact is the §3.2 campaign-wide prompt
snapshot. Until `snapshot-v3-prompts` is invoked, operators may edit the prompt
templates. That command copies one common ordinary/assumption bundle, adds the
v3 feasibility-and-falsification clause, hashes it, places the same hash in
every run manifest, and enables launch. Later prompt experiments use another
campaign bundle.

Every v3 proposal otherwise uses a fresh bounded conversation and a
deterministic state capsule. Raw transcripts are retained for audit but never
replayed. The capsule includes current public evidence and a bounded,
deterministically selected set of recent, successful, and informative failed
outcomes. It omits internal condition labels, horizons, remaining resources,
selection counts, private evaluator fields, and experiment language.

The assumption-changing text remains open-ended. It adds evidence-for/against,
feasibility, and falsification guidance without a mechanism menu. At matched
intervention indices, v3 exports blinded source-delta packets for C0–C3 with
the manipulation-check fields described in §4.3.

## Flexible runtime and research options

`inputs/v3-runtime.json` contains revisable controls for:

- context/capsule size and evidence selection;
- intervention variants and cadence experiments;
- portfolio estimands, capacity/selection/retention/source variants;
- task-isolated evaluator capacity, repeated training seeds, confirmation, and
  multi-fidelity task plugins;
- ecological Autoresearch, native OpenEvolve, model, and task companions; and
- instrumentation, review, semantic diagnostics, and analysis outputs.

The controller reads this file at each opportunity. `update-v3-runtime` replaces
it with a complete append-only before/after receipt and a human reason. These
settings are not included in the immutable prompt-bundle mechanism. A task
extension may implement `prepare_seed_workspace` and `make_evaluator`; a custom
framework supplies `adapter_factory="module:callable"`. Examples are under
`configs/tasks/` and `configs/frameworks/`.

## Implemented improvement map

- §3.1: task-specific evaluator pools, one host safety ceiling, queue time
  outside candidate runtime, contention receipts, configurable concurrency,
  and health telemetry. Remote workers remain available for timing-sensitive
  objectives.
- §3.2: one deliberate campaign prompt snapshot and launch gate.
- §3.4: stable remote call IDs, pre-dispatch receipts, payload/result hashes,
  worker-side result caches, a retrieval attempt after response loss, and
  exactly-once scientific charging.
- §3.5: prompt/input/support/disk/clock/accelerator/tensor/pairing health checks
  plus disposable crash-recovery tests.
- §4: the open intervention, feasibility/falsification addition, matched
  blinded manipulation packets, exploration-versus-retention records, and
  configurable alternative intervention variants.
- §5: explicit whole-system or matched-context estimands, bounded evidence,
  configurable branch/population variants, and lineage/selection/source
  diagnostics.
- §6: bounded Autoresearch sessions, deterministic capsules, controller-side
  preflight, ordinary Git workspaces, flexible prose metadata, source/IR
  provenance, transcript retention for audit, and continuous ecological
  companions through existing adapters.
- §7: bounded OpenEvolve calls, informative evidence selection, free-form
  mechanism labels plus semantic-delta fingerprints, source validity separated
  from missing prose metadata, atomic patches, and native/population companion
  extension points.
- §8: repeated training-seed aggregation, public/private/Layer-C task hooks,
  existing addition anti-solver checks, nanoGPT worker telemetry, Fashion-MNIST
  task isolation, and a task-evaluator plugin path for multi-fidelity designs.
- §9: the supplied preset uses more 100-proposal blocks, individual paired
  launches, condition-blind task queues, overlap receipts, and arbitrary
  task/model strata through the same protocol.
- §10: source-authoritative blinded packets, mechanism-quality annotation
  fields, trajectory-level summaries, raw run outputs, paired-prefix contrasts,
  and explicit non-independence of proposals.
- §11: per-item raw Codex logs, per-proposal token components, revisable pricing,
  physical shared-prefix accounting, queue/evaluator/lifecycle time receipts,
  source diffs, architecture IR, environment hashes, health, audit, data
  dictionary, and replayable events.
- §12: artifact-clean subject boundaries, bounded concise prompts, sanitized
  failures, and no new mechanism restrictions.
- §13: both recommended successor bundles are expressible through the one v3
  protocol and framework adapters rather than separate protocol versions.

## Deliberately omitted immutability/restriction clauses

Per the operator's instruction, v3 does not add the following restrictions:

- §3.1.1's requirement to freeze the selected concurrency and §3.1.5's freeze
  of CPU priority, power, and thermal policy;
- all of §3.3 (model/service/tool/CLI settings remain observable and revisable);
- §4.4.3–§4.4.5's fixed/jittered/non-adaptive scheduling requirements and the
  frozen wording-library portion of §4.6;
- the frozen evidence/capsule/session/selector limits in §5.2, §6.1–§6.3, and
  §7.2, plus the prospectively fixed metadata choice in §7.4;
- the pinned/frozen companion and crossover language in §7.6–§7.7;
- frozen seed/margin/promotion/view/validation/holdout-set requirements in
  §8.1 and §8.4–§8.6;
- the separately frozen horizon in §9.1 and all prescriptive stopping rules in
  §9.2;
- rubric/model/missing-data preregistration requirements in §10.1 and §10.4;
  and
- §11.5's requirement to freeze an archive manifest.

The corresponding capabilities remain configurable and auditable; they simply
are not made immutable or mandatory.

## Commands

Use one protocol for either framework:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/unified_v3.toml
```

After calibration and campaign creation, inspect/edit the prompt sources, then:

```bash
$PY -m $CLI snapshot-v3-prompts --campaign "$OUT-campaign"
$PY -m $CLI validate --campaign "$OUT-campaign"
$PY -m $CLI v3-health --campaign "$OUT-campaign"
$PY -m $CLI run-v3-one --campaign "$OUT-campaign" --run-id "$RUN_ID" --python-bin "$PY"
$PY -m $CLI audit-v3 --campaign "$OUT-campaign"
$PY -m $CLI summarize-v3 --campaign "$OUT-campaign"
$PY -m $CLI manifest-v3-release --campaign "$OUT-campaign" --output release.json
```

`start-staged-trajectory`, `pause-staged-trajectory`, and
`resume-staged-trajectory` also understand v3. Before the fork, either member's
controller advances the one pair-owned prefix under a pair lock. After the
fork, each run is independently controllable.
