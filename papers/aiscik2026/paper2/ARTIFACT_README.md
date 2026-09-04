# State-matched autonomous-research intervention artifact

This archive reproduces the analyses reported in *Can a Prompt Defixate an
Autonomous Compression Agent? State-Matched Forks in Model Search*.

## Contents

- `papers/aiscik2026/paper2/analysis.py`: complete deterministic analysis
  source (fixed seed `20260901`).
- `papers/aiscik2026/paper2/derived/`: proposal-level tables, exact-fork and
  trajectory pair tables, bootstrap summaries, exploratory mechanism-family
  coding, Fashion-MNIST portability tables, and publication figures.
- `data/c0c3/unified-v3-tiny-adderboard-greedy-campaign/`: the 32 primary
  greedy OpenEvolve trajectories, protocol/task/framework records, manifests,
  completed-event streams, proposal-10 prompt snapshots, proposal final
  messages, candidate provenance, and source snapshots used by the analysis.
- `data/c0c3/unified-v3-tiny-adderboard-native-campaign/`: the corresponding
  32 native OpenEvolve trajectories and analysis-required source records.
- `data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/`: the 20 complete
  trajectories used only for descriptive cross-task replication.
- `PAPER2_SHA256SUMS`: SHA-256 digest for every payload file in the archive.
- `LICENSE`: MIT license for the included analysis and protocol source.

## Reproduction

Unpack the archive, install NumPy and Matplotlib using the recorded versions,
and run from the archive root:

```bash
MPLCONFIGDIR=/tmp/aiscik-mpl python3 \
  papers/aiscik2026/paper2/analysis.py
```

The script first validates 64 primary trajectories, 32 matched forks, 288
identical paired-prefix proposal events, matching fork parents, insertion-only
treatment prompts, 4,480 common-horizon events, and 4,480 recorded final
proposal messages. It then recreates every CSV, JSON summary, and figure. No
network access or model call is required.

## Analysis boundaries

Opportunities 1-9 are shared within each C0/C1 and C2/C3 pair. Opportunity 10
is the exact state-matched fork. Opportunities 10-19 form the clean
one-intervention phase. The repeated-treatment analysis stops at opportunity
70, selected mechanically as the minimum completed horizon across all primary
trajectories when the corpus was frozen, without consulting outcomes.
Fashion-MNIST trajectories are already divergent at intervention checkpoints
and are therefore labeled a descriptive portability check rather than
exact-fork evidence.

The artifact reports all outcomes, including failed and unretained proposals.
Source and text novelty metrics are transparent operational proxies, not
validated measures of latent cognition or globally new science.

## Privacy and exclusions

The archive contains no API keys, provider credentials, user names, absolute
host paths, repository remotes, private chain-of-thought, or dashboard-control
state. Machine-local locks, provider event streams, bytecode, evaluator caches,
large duplicate workspaces, redundant per-opportunity prompt transcripts, and
other files not read by the analysis are excluded.
