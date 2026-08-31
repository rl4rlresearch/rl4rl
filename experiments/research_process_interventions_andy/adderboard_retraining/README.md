# AdderBoard incumbent retraining

This subexperiment retrains the final incumbent from each of the 24 completed
research trajectories. Identical incumbents are deduplicated, yielding 18
versioned architecture snapshots in `candidates/`.

This is supervised training of the discovered neural networks on addition
examples. It is not reinforcement learning of the research-agent policy.

## Existing results

- [`reports/1000_step_screen.md`](reports/1000_step_screen.md): all 18 candidates,
  1,000 steps each, with dense diagnostic metrics.
- [`reports/5000_step_official.md`](reports/5000_step_official.md): all 18
  candidates, 5,000 steps each, evaluated with the unmodified vendored
  AdderBoard verifier.

The 5,000-step study found no qualifying architecture. The tracked Modal
launcher now defaults to this 5,000-step budget, and its compatibility
`final` stage is capped at the same budget. The historical 1,000-step screen
must be selected explicitly.

## Validate and plan

From the repository root:

```bash
MODAL_PROFILE=scalingintelligence MODAL_ENVIRONMENT=main \
  architecture_discovery/.venv/bin/modal run \
  experiments/research_process_interventions_andy/adderboard_retraining/modal_retrain.py \
  --action plan --stage develop
```

The plan action starts no remote training. A launch additionally requires both
`--approved` and `RL4RL_RETRAIN_APPROVED=YES`.

## Rebuild candidate snapshots

`prepare_candidates.py` reads the checked-in dashboard data and the verified
download bundles under `architecture_discovery/outputs/development/modal_downloads`.
It refuses to overwrite a nonempty candidate directory.

```bash
architecture_discovery/.venv/bin/python \
  experiments/research_process_interventions_andy/adderboard_retraining/prepare_candidates.py \
  --output /tmp/rl4rl-candidate-rebuild
```

Compare the resulting manifest and file hashes with `candidates/` before
replacing any tracked snapshot. Raw checkpoints and Modal volume downloads are
intentionally not committed.
