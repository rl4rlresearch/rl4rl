# Greedy OpenEvolve transformer protocols v2.0 and v2.1

Protocol 2.0 is a prospective, scientifically separate Greedy OpenEvolve stratum. It
keeps the C0–C3 2×2 question and removes N0 entirely. It must not be pooled with
the legacy protocol-1.1 Greedy OpenEvolve campaign, whose task surface allowed direct
arithmetic transducers and whose 30,000-step failures consumed many hours.
Protocol 2.1 is the prospective source-only, artifact-clean successor. It keeps
the same execution geometry and validity checks but removes unnecessary
controller information and prompt duplication. Use 2.1 for new artifact-clean
collection; existing 2.0 campaigns remain scientifically separate.

## Frozen design

- V2.0 uses three paired blocks; the v2.1 addition preset uses five. Both have
  four C0–C3 trajectories per block and no N0 assignment.
- `K=4`, the existing fair lineage selector, and strict lineage-local retention.
- 200 proposal opportunities per trajectory, with C1/C3 assumption changes at
  every tenth opportunity.
- GPT-5.6 Sol xhigh for every proposal.
- A fresh bounded Codex call per proposal. Cross-proposal continuity is the
  selected source, all retained designs, the last twelve outcomes, and a compact
  free-form mechanism ledger in v2.0; v2.1 records mechanisms once in its recent
  outcomes and has no duplicate ledger. Raw conversation history is not resumed.
- One independently supervised process per trajectory. All processes may
  generate proposals concurrently. V2.0 retains three campaign evaluator slots;
  v2.1 uses one per declared block. All local campaigns share twelve host slots.
- The 1,644-parameter pair-token transformer is the common seed. Its ordinary
  training path is 5,000 steps rather than the legacy 6,080-parent 30,000-step
  path. The evaluator timeout is 1,800 seconds.

## Subject and model validity

The subject sees an ordinary job to minimize a trained 10-digit-addition
transformer. It does not see benchmark, study, treatment, condition, or C0–C3
labels. Only `src/model.py` and `src/train.py` are editable. Data processing,
generic decoding, and verification are protected.

Every patch must retain a genuinely learned autoregressive transformer. Before
training, deterministic source preflight rejects syntax errors, recognized
arithmetic/transducer shortcuts, missing files, duplicate source, malformed
patches, unmatched patch blocks, and ambiguous patch blocks. Candidate training
starts with no checkpoint and must leave both a best checkpoint and a
positive-step last checkpoint. After training, verification rejects source
mutation, zero learned state, absent or unused self-attention, and models whose
accuracy remains high when attention is ablated. Layer A uses the fixed seed
2025 set; sealed Layer C uses disjoint seed 8,724,319.

V2.1 receives source and verified public metrics but never the supplied trained
checkpoint. It also receives no proposal/horizon/resource counters, prompt-slot
markers, empty design placeholders, selection counts, nonpublic `cases` or
`correct` fields, raw runner errors, filesystem paths that do not exist, or
run-derived environment seed. See `ARTIFACT_CLEAN_PROTOCOLS.md`.

## Greedy OpenEvolve boundary

This is the Greedy OpenEvolve proposal adapter, not native end-to-end
OpenEvolve. The vendored `PromptSampler`, SEARCH/REPLACE representation, parser,
and patch workflow remain. The shared C0–C3 instrument deliberately owns parent
selection, portfolio state, retention, budgets, and evaluator feedback because
those are the randomized factors. The v2 sampler includes the selected source
once and every alternate retained source once, avoiding the legacy prompt's
duplicate previous/top/inspiration copies.

The subject supplies a free-form mechanism name, falsifiable hypothesis,
intended edit, and evidence citation. No fixed mechanism-family menu is shown.
Those fields provide auditable provenance and a bounded ledger without
restricting which mechanisms may be proposed.

In v2.1 the current/reference source, public design evidence, metadata contract,
and SEARCH/REPLACE contract each appear once. V2.0's repeated mechanism ledger,
metrics, and adapter-appended response reminder are not rendered.

## Failure and speed controls

- Every started proposal is charged; no invalid candidate is retried or erased.
- Provider and remote-infrastructure failures are distinguished from malformed
  patches, source preflight, model-contract failure, timeout, execution failure,
  and nonqualification.
- Deterministic patch/source failures consume no evaluator call.
- Training runs at most 30 minutes, preventing the prior multi-hour 0%-accuracy
  failures. Training and Layer-A verification remain separate: the editable
  trainer selects on its protected validation split, then the trusted verifier
  decides qualification.
- Local MPS and Modal L4 are different hardware strata. Choose one task TOML,
  calibrate it on that backend, and never change backend inside a campaign.

## Local MPS setup (do not launch until ready)

From the repository root:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial
PROTOCOL=$C0C3/configs/protocols/controlled_openevolve_transformer_v2.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_mps.toml
FRAMEWORK=$C0C3/configs/frameworks/openevolve_v2.toml
OUT=data/c0c3/controlled-openevolve-transformer-v2-mps

export ADDERBOARD_CODEX_1644_SOURCE='/path/to/frozen/codex-1644-source'

$PY -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'

$PY -m $CLI calibrate \
  --protocol "$PROTOCOL" --task "$TASK" \
  --output "$OUT-calibration" --python-bin "$PY"

$PY -m $CLI create \
  --protocol "$PROTOCOL" --task "$TASK" --framework "$FRAMEWORK" \
  --baseline "$OUT-calibration/baseline.json" --output "$OUT-campaign"

$PY -m $CLI validate --campaign "$OUT-campaign"
```

Do not create an MPS calibration unless that preflight succeeds in the same
terminal/runtime that will execute the campaign. An unavailable MPS device is
an infrastructure failure, not a candidate result. Use the separately
calibrated Modal L4 task below, or define and preregister another hardware
stratum, rather than silently falling back inside an MPS campaign.

Codex's restricted command sandbox can hide Metal even on a supported Apple
Silicon Mac. A false result inside that sandbox does not mean PyTorch is
misinstalled. Run the probe and the detached supervisor from an ordinary VS
Code/macOS terminal. The `openevolve-v2` supervisor profile repeats a real
tensor-allocation probe during both `check` and `start`, so a process that
cannot access the frozen accelerator fails before any proposal is consumed.

Do not pass `--without-no-search`: protocol 2.0 already freezes N0 as absent,
and campaign creation records that composition in its hashes and manifest.

For v2.1, replace the three frozen inputs and output prefix above with:

```bash
PROTOCOL=$C0C3/configs/protocols/controlled_openevolve_transformer_v2_1.toml
TASK=$C0C3/configs/tasks/ten_digit_addition_pair_transformer_openevolve_v2_1_mps.toml
FRAMEWORK=$C0C3/configs/frameworks/openevolve_v2_1.toml
OUT=data/c0c3/controlled-openevolve-transformer-v2-1-mps
```

Use the `openevolve-v2.1` supervisor profile and detached runtime path shown in
`RUNBOOK.md`. V2.1 also freezes N0 as absent.

Create a detached runtime from the exact committed launch revision, then use
the durable supervisor:

```bash
git worktree add --detach /private/tmp/rl4rl-c0c3-openevolve-v2 HEAD

RL4RL_OVERNIGHT_PROFILE=openevolve-v2 \
  architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py check

RL4RL_OVERNIGHT_PROFILE=openevolve-v2 \
  architecture_discovery/.venv/bin/python experiments/c0c3_overnight.py start \
  --recover-interrupted --all-running
```

Use the profile-specific `status`, `pause`, `resume`, and `stop` commands. The
supervisor never owns another profile's processes or campaign.

## Optional Modal L4 evaluator

The Modal variant keeps every Codex call local and sends only task support plus
one candidate snapshot to a deployed evaluator. It needs no OpenAI secret. The
lab owner should create a dedicated Modal environment with a conservative hard
budget, authenticate the local CLI/profile, and deploy the evaluator once:

```bash
MODAL=architecture_discovery/.venv/bin/modal
$MODAL deploy -m experiments.c0c3_factorial.modal_hybrid_app
```

Then repeat calibration/campaign creation with
`ten_digit_addition_pair_transformer_openevolve_v2_modal.toml`. The deployed
function permits at most three L4 containers and retains warm containers for at
most five idle minutes. Do not reuse an MPS calibration or campaign.

Campaign-attributed GPU time is available without mutating the campaign:

```bash
$PY -m $CLI modal-usage --campaign "$OUT-campaign"
$MODAL billing --help
```

`modal-usage` counts completed/failed remote calls and recorded worker seconds.
It is not an invoice and does not include all warm-idle, storage, credits, or
workspace-wide use. Use Modal's Usage & Billing page or `modal billing` for the
authoritative account balance and set the workspace/environment budget before
launch.

For v2.1 Modal collection, use
`ten_digit_addition_pair_transformer_openevolve_v2_1_modal.toml`; do not reuse a
v2.0 or MPS calibration.
