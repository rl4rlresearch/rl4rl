# RL4RL: Autonomous Research Trajectories

This branch studies **how autonomous research systems search**, using their
complete AdderBoard model-compression trajectories rather than only their final
leaderboard entries. The target is the smallest valid model reaching at least
99% exact-match accuracy; the research outcome is the strategy the agent used
to get there.

The primary comparison is among OpenEvolve, Autoresearch, and TTT-Discover. The
analysis asks whether they mainly make local, ontology-preserving edits—width,
depth, normalization, and weight tying—or cross architecture boundaries with
new representations, positional mechanisms, algebraic projections, or other
ontology-changing moves.

Start with:

- [`TRAJECTORY_STUDY_PROTOCOL.md`](TRAJECTORY_STUDY_PROTOCOL.md) — active
  scientific protocol and interpretation limits.
- [`architecture_discovery/README.md`](architecture_discovery/README.md) —
  executable workflow, schemas, adapters, and output contract.
- [`architecture_discovery/trajectory_study_manifest.template.yaml`](architecture_discovery/trajectory_study_manifest.template.yaml)
  — frozen-input manifest template.
- [`architecture_discovery/trajectory_annotation_codebook.yaml`](architecture_discovery/trajectory_annotation_codebook.yaml)
  — double-coding rules.

## Quick offline validation

```bash
git submodule update --init --recursive
cd architecture_discovery
uv sync --python 3.12
.venv/bin/python -m pytest -q
.venv/bin/python scripts/trajectory_offline_smoke.py \
  --output-dir /private/tmp/rl4rl-trajectory-smoke
```

The smoke command uses explicitly synthetic data and cannot be mistaken for a
scientific result. Real analysis remains data-blocked until complete trajectory
exports and human annotations are added to a hash-frozen manifest.

The previous C0–C3 prospective architecture-novelty apparatus remains in the
repository for provenance and reuse, but it is not the active protocol on this
branch. Its historical plans are clearly marked as superseded.
