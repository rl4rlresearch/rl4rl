# Trace-level construct-validity audit artifact

This archive reproduces the analyses reported in *What Does an Autonomous
Research Benchmark Measure? A Trace-Level Construct-Validity Audit of 4,000
Agent Proposals*.

## Contents

- `papers/aiscik2026/paper1/analysis.py`: complete analysis source (fixed seed
  `20260901`).
- `papers/aiscik2026/paper1/derived/`: the complete proposal-level table,
  run/scope/condition/phase/edit
  summaries, bootstrap results, qualitative sample manifest and reader, and
  publication figures.
- `data/c0c3/fashion-mnist-openevolve-v2-1-mps-campaign/`: campaign, protocol,
  task, framework, schedule, validation, amendment records, every trajectory's
  manifest/state/lifecycle/event log, all candidate source snapshots, and all
  raw final messages read by the analysis.
- `PAPER1_SHA256SUMS`: SHA-256 digest for every file in the archive except the digest
  file itself.
- `LICENSE`: MIT license for the included source and analysis code.

## Reproduction

Unpack the archive, install NumPy and Matplotlib (versions used are recorded in
`papers/aiscik2026/paper1/requirements.txt`), and run from the archive root:

```bash
MPLCONFIGDIR=/tmp/aiscik-mpl python3 \
  papers/aiscik2026/paper1/analysis.py
```

The script asserts 20 trajectories, 200 completed proposals per trajectory,
and 4,000 proposal rows. It reads only the recorded campaign and requires no
network access or model calls.

## Scope and privacy

Blocks 1-3 are the original campaign scope. Blocks 4-5 are the recorded
operator-authorized extension and are reported separately in the paper. The
archive contains no API keys, provider credentials, user names, absolute host
paths, or repository remotes. Machine-local lock files, controller state,
provider event streams, caches, and redundant evaluator workspaces are excluded.
