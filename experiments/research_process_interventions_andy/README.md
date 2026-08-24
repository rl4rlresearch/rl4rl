# Research-process interventions

This folder contains the complete experiment layer for studying how portfolio
memory and assumption-challenge interventions change an autonomous research
agent's search trajectory. The primary outcomes are process measures—idea
lineages, regime changes, revisitation, structural diversity, invalid proposals,
and parameter-count frontiers—not only the final architecture.

## Contents

- `run.py`: CLI entry point for planning and validating intervention studies.
- `configs/example.json`: example factorial-study configuration.
- `dashboard/`: interactive visualization and its checked-in 24-run dataset.
- `reports/`: pilot and 8/12/20-horizon trajectory reports.
- `adderboard_retraining/`: frozen final architectures, isolated Modal training
  launcher, and the 1,000- and 5,000-step evaluation reports.

The reusable controller integrations and causal logging implementation remain
under `architecture_discovery/research_dynamics/`. They are shared runtime code,
not a single experiment artifact, and moving them would break the controller and
Modal import boundaries.

## Plan an intervention study

From the repository root:

```bash
architecture_discovery/.venv/bin/python \
  experiments/research_process_interventions_andy/run.py --help
```

For a concrete full-trajectory planning example, see
[`architecture_discovery/README.md`](../../architecture_discovery/README.md).
Planning and local validation do not start paid provider or Modal work.

## Inspect the completed sweep

The dashboard dataset contains 24 runs: two frameworks, four intervention
conditions, and horizons of 8, 12, and 20 proposals. Start it with:

```bash
cd experiments/research_process_interventions_andy/dashboard
npm install
npm run dev
```

Read [`reports/architecture_trajectories_8_12_20.md`](reports/architecture_trajectories_8_12_20.md)
for the corresponding static report.

## Data policy

The compact, immutable candidate JSON files and summarized results are tracked.
Raw Modal checkpoints, logs, and downloaded result bundles remain under ignored
`architecture_discovery/outputs/` or operator-selected download directories.
No API keys or Modal credentials belong in this folder.
