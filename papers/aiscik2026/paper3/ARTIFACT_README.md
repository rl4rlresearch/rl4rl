# Population-memory lineage-audit artifact

This archive reproduces the analyses reported in *Pluralism Without a Free
Lunch: A Lineage Audit of Population Memory in Autonomous ML Research*.

## Contents

- `papers/aiscik2026/paper3/analysis.py`: deterministic analysis source,
  including the fixed block-bootstrap seed `20260901`.
- `papers/aiscik2026/paper3/derived/`: trajectory metrics, paired contrasts,
  system summaries, source-audited alternative-branch examples, figures,
  aggregate metadata, and input-file hashes.
- `data/c0c3/unified-v3-tiny-adderboard-greedy-campaign/`: 32 Tiny
  AdderBoard trajectories using the deterministic K=1/K=4 greedy controller.
- `data/c0c3/unified-v3-tiny-adderboard-native-campaign/`: 32 corresponding
  trajectories using the vendored native OpenEvolve population database,
  including the native population event ledgers.
- `data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/`: 20 complete
  Fashion-MNIST trajectories used for descriptive cross-task replication.
- `PAPER3_SHA256SUMS`: SHA-256 digest for every payload file.
- `LICENSE`: MIT license for the complete artifact payload, including code,
  trace records, candidate source snapshots, data, and documentation.

The event records include the agent-authored mechanism, hypothesis, intended
edit, and evidence summary for every analyzed proposal. Candidate source
snapshots include the baseline, generated candidates, and selected parents
required by the parent graph. Opportunity-10 prompt snapshots make the
subject-visible memory composition auditable. Native event ledgers expose
parent sampling, inspiration programs, admission, archive, and island state.

## Reproduction

Unpack the archive, install the two pinned dependencies, and run from the
archive root:

```bash
MPLCONFIGDIR=/tmp/aiscik-p3 python3 \
  papers/aiscik2026/paper3/analysis.py --data-root . --verify-input-hashes
```

The script verifies the frozen 284-file raw-input hash ledger, then validates
64 Tiny runs, 20 Fashion-MNIST runs, contiguous proposal horizons, and the
native population ledgers before regenerating every analytic table and figure.
It uses the mechanically selected common Tiny horizon of 70 and all 200 Fashion
opportunities. No network, model, evaluator, or training call is required.

Before reproduction, verify every packaged payload byte from the archive root:

```bash
shasum -a 256 -c PAPER3_SHA256SUMS
```

## Interpretation boundaries

The primary K=1 versus K=4 comparisons use only the greedy controller and are
paired by block within prompt policy. They estimate a whole-system portfolio
contrast, including the extra evidence shown in K=4 prompts. The native
OpenEvolve adapter supersedes the generic controller's nominal memory factor;
native C0-C3 labels are therefore not interpreted as K=1/K=4 assignments.

The trajectory is the unit of analysis. Bootstrap intervals resample blocks
and are descriptive sensitivity ranges, not randomization-based confidence
intervals. Source examples establish that dormant branches remained reachable
and later recorded strict incumbent improvements; they are not estimates of
average causal benefit. The strict reference-attribution rule matches the
literal phrases `reference design`, `alternative design`, and `available
design`; a broader `reference` rule is retained as a sensitivity analysis.
Both measure observable reporting, not private reasoning.

## Privacy and exclusions

The archive contains no credentials, repository remotes, user names, absolute
host paths, provider event streams, private chain-of-thought, dashboard state,
locks, evaluator caches, or duplicate workspaces. Machine-local strings in
included text records are replaced with anonymous placeholders. The artifact
does not contain the paper PDF.
