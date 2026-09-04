# Fashion-MNIST artifact-clean task

This is a third, local-Mac task stratum for protocols 1.7 and 2.1. It mirrors
the nanoGPT research-system geometry while replacing fixed-time H100 language
modeling with fixed-exposure MPS image classification.

Nothing in this setup command sequence starts a trajectory implicitly.

## Frozen task contract

- Input: 28×28 grayscale images and ten class labels.
- Candidate source: only `train.py`.
- Candidate-train split: a fixed seeded 50,000-image subset of the official
  60,000-image training set.
- Public validation: the other fixed 10,000 official training images.
- Layer C: the untouched official 10,000-image test set.
- Training exposure: exactly 100,000 presented examples (two passes over the
  candidate-train split), independent of batch size.
- Preprocessing: fixed scalar normalization owned by the evaluator.
- Parameter ceiling: 250,000 learned parameters. The supplied baseline has
  105,866 parameters, leaving room for substantially different small models
  without making the task compute-heavy.
- Hard evaluator timeout: 90 seconds. This is a failure guard, not the
  optimization objective.
- Objective: lexicographically maximize exact public-validation correct count,
  then minimize public-validation cross-entropy on exact-count ties.

The controller requires one scalar fitness, so the evaluator encodes the exact
lexicographic ordering as:

```text
validation_score = validation_correct + 0.5 / (1 + validation_cross_entropy)
```

The fractional term is always at most 0.5, so it can never compensate for one
fewer correct prediction. The public output also reports the unencoded correct
count, accuracy, cross-entropy, parameters, examples, optimizer steps, training
seconds, and batch size.

The protected loop owns data access, presentation order, counting, backward,
one optimizer step per presented batch, validation, and scoring. Candidate
`train.py` owns the architecture, optimizer, training loss, augmentation hook,
batch size, clipping choice, and schedule hook. Every verification starts from
a fresh initialization. The sealed test set is never materialized in the
subject workspace or used for online selection.

## Prepared research-system strata

Autoresearch v1.7 matches nanoGPT's four-block setup: 16 independent C0–C3
trajectories, 200 proposals each, one persistent Codex conversation per run,
and an assumption-changing instruction every tenth proposal.

Greedy OpenEvolve v2.1 matches nanoGPT's three-block setup: 12 independent C0–C3
trajectories, 200 bounded ephemeral proposals each, and the same intervention
schedule. Neither stratum contains N0. Both use GPT-5.6 Sol xhigh in Fast mode
and the shared 12-slot local evaluator scheduler. The campaign-local ceilings
are four and three trainers respectively because protocols 1.7 and 2.1 use one
slot per declared block.

## Prepare later (network and evaluation begin here)

Run from the repository root. The first command is the only dataset download:

```bash
PY=architecture_discovery/.venv/bin/python
CLI=experiments.c0c3_factorial.cli
C0C3=experiments/c0c3_factorial

$PY -m experiments.c0c3_factorial.fashion_mnist prepare --repo-root .
```

The downloader verifies the four official Fashion-MNIST IDX checksums and
writes ignored data under `data/raw/fashion-mnist/`. Inspect
`data/raw/fashion-mnist/manifest.json` before calibration.

Calibrate each protocol separately because calibration records the protocol
hash. These commands do train the supplied baseline and have intentionally not
been run during setup:

```bash
TASK=$C0C3/configs/tasks/fashion_mnist_source_only_mps.toml

AR_PROTOCOL=$C0C3/configs/protocols/fashion_mnist_autoresearch_v1_7.toml
AR_FRAMEWORK=$C0C3/configs/frameworks/autoresearch_fashion_mnist_v1_7.toml
AR_OUT=data/c0c3/fashion-mnist-autoresearch-v1-7-mps
$PY -m $CLI calibrate --protocol "$AR_PROTOCOL" --task "$TASK" \
  --output "$AR_OUT-calibration" --python-bin "$PY"
$PY -m $CLI create --protocol "$AR_PROTOCOL" --task "$TASK" \
  --framework "$AR_FRAMEWORK" --baseline "$AR_OUT-calibration/baseline.json" \
  --output "$AR_OUT-campaign"
$PY -m $CLI validate --campaign "$AR_OUT-campaign"

OE_PROTOCOL=$C0C3/configs/protocols/fashion_mnist_openevolve_v2_1.toml
OE_FRAMEWORK=$C0C3/configs/frameworks/openevolve_fashion_mnist_v2_1.toml
OE_OUT=data/c0c3/fashion-mnist-openevolve-v2-1-mps
$PY -m $CLI calibrate --protocol "$OE_PROTOCOL" --task "$TASK" \
  --output "$OE_OUT-calibration" --python-bin "$PY"
$PY -m $CLI create --protocol "$OE_PROTOCOL" --task "$TASK" \
  --framework "$OE_FRAMEWORK" --baseline "$OE_OUT-calibration/baseline.json" \
  --output "$OE_OUT-campaign"
$PY -m $CLI validate --campaign "$OE_OUT-campaign"
```

Before starting any trajectory, confirm both baseline records report exactly
100,000 examples, no more than 250,000 parameters, plausible accuracy, and less
than 90 seconds of evaluator wall time. A timeout or a result uncomfortably
close to 90 seconds is a calibration failure; adjust a new prospective preset
before collecting proposal data.

## Durable launch commands (do not run during setup)

Create isolated runtimes from the final committed revision, then use the two
independent supervisor profiles:

```bash
git worktree add --detach /private/tmp/rl4rl-c0c3-autoresearch-v1-7-fashion-mnist HEAD
git worktree add --detach /private/tmp/rl4rl-c0c3-openevolve-v2-1-fashion-mnist HEAD

RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7-fashion-mnist \
  $PY experiments/c0c3_overnight.py check
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-fashion-mnist \
  $PY experiments/c0c3_overnight.py check

# These are the actual launch commands; run them only when ready.
RL4RL_OVERNIGHT_PROFILE=autoresearch-v1.7-fashion-mnist \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
RL4RL_OVERNIGHT_PROFILE=openevolve-v2.1-fashion-mnist \
  $PY experiments/c0c3_overnight.py start --recover-interrupted --all-running
```

The supervisors point detached evaluators to the checksum-verified dataset in
the main checkout through `RL4RL_FASHION_MNIST_DATA_ROOT`. Override that value
only before calibration and campaign creation if a different stable absolute
cache is required. The assumption-changing files remain operator-editable only
until each trajectory's first start, exactly as in the other artifact-clean
strata.

The live dashboard has dormant sections for both prospective campaign paths.
They remain unavailable until campaigns are created and begin producing logs.
