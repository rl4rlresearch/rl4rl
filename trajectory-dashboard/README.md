# RL4RL Architecture Trajectory Dashboard

Interactive dashboard for the 24-run architecture-search sweep:

- AutoResearch and OpenEvolve
- 8-, 12-, and 20-proposal horizons
- sequential or portfolio memory
- neutral review or assumption-challenge deliberation

The dashboard compares proposal number against parameter count, exposes every
architecture and failure event, and clearly separates structural validity from
AdderBoard exact-match accuracy.

## Data

The checked-in dataset is generated from independently verified Modal artifact
bundles. Rebuild it from downloaded artifacts with:

```bash
python3 scripts/build_trajectory_assets.py \
  --downloads-root ../architecture_discovery/outputs/development/modal_downloads \
  --output-json app/data/trajectories.json \
  --report ../ARCHITECTURE_TRAJECTORIES_8_12_20.md
```

The generator requires the complete 24-cell matrix and exactly 320 proposal
opportunities unless `--allow-partial` is explicitly supplied.

## Development

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

Validation:

```bash
npm run lint
npm test
```

`npm test` performs a production build, checks the rendered dashboard, and
verifies the exact 24-run/320-proposal data matrix.

## Interpretation

These runs used a 10-step engineering-smoke training profile. Structural
validity means an architecture executed and passed the runtime graph checks; it
does not imply task competence. Every evaluated proposal had zero full-answer
exact-match accuracy, so very small parameter-count incumbents are evidence of
objective collapse rather than successful AdderBoard models.
