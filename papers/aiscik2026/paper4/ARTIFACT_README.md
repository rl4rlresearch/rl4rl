# Paper 4 Reproducibility Artifact

This archive reproduces the analysis for:

**Forgetting Without Restarting: Periodic History Refresh in Autonomous ML
Research Agents**

The artifact contains only already-recorded experiment logs and source
snapshots. It does not call any model provider, launch any evaluator, or train
any candidate.

## Contents

- `papers/aiscik2026/paper4/analysis.py`: the complete analysis script.
- `papers/aiscik2026/paper4/derived/`: reproduced CSV, JSON, and figure
  outputs.
- `data/c0c3/...`: the focal passive-control and periodic-full-refresh run
  records used by the paper, filtered to the analyzed horizons.
- `PAPER4_SHA256SUMS`: SHA-256 checksums for every packaged payload file.
- `LICENSE`: MIT license for the packaged artifact payload.

The four analyzed strata are:

- Fashion-MNIST, greedy controller, common horizon 43.
- Fashion-MNIST, native OpenEvolve controller, common horizon 13.
- Tiny Addition, greedy controller, common horizon 92.
- Tiny Addition, native OpenEvolve controller, common horizon 50.

In each stratum there are three matched passive-control trajectories and
three matched periodic-full-refresh trajectories. Within every replicate, both
arms share proposals 1-5 exactly and fork at proposal 6. The refresh arm then
clears subject-visible search history every five proposals while retaining the
verified incumbent model.

## Reproduction

From the unpacked archive root:

```bash
shasum -a 256 -c PAPER4_SHA256SUMS
python3 -m venv .venv
.venv/bin/python -m pip install -r papers/aiscik2026/paper4/requirements.txt
.venv/bin/python papers/aiscik2026/paper4/analysis.py --data-root . --verify-input-hashes
```

`PAPER4_SHA256SUMS` verifies the packaged payload as released. Run it before
regenerating derived outputs, because Matplotlib PDF/PNG metadata can differ
after a fresh render. The `--verify-input-hashes` gate checks the raw inputs
against the frozen input ledger before writing derived outputs. The numerical
CSV/JSON outputs are deterministic under the included script, and the script
uses a fixed bootstrap seed of `20260901`.

## Claim Boundary

The campaigns were paused before their advertised 200-proposal endpoint. The
paper therefore analyzes the largest contiguous common horizon available within
each task-by-controller stratum and treats every result as right-censored. The
trajectory pair, not the proposal, is the replication unit.

Agent messages are analyzed only as recorded final proposal summaries. They
are not private chain-of-thought and are not treated as direct evidence of
hidden cognition.
