# Frozen parent seeds

## Codex 1,644-parameter pair-token parent

This is a separate v1.5 stratum from the 6,080-parameter standard-token parent.
It must not be pooled with that stratum as though the starting representation
were identical.

- Upstream source: `https://github.com/anadim/smallest-addition-transformer-codex`
- Frozen upstream commit: `9de34061c8c268e0fc2198e473911a62b152254c`
- Published architecture: one-layer decoder-only transformer with pair-column
  input tokens, `d_model=8`, `n_head=2`, `d_ff=12`, and tied embeddings.
- Parent calibration under this repository's pinned AdderBoard verifier:
  1,644 deduplicated parameters; 9,962/10,010 correct (99.52%) at seed 2025.

The task adapter copies only the necessary model/data/training/evaluation source
files plus the checkpoint into the subject workspace. This provenance note is
not copied there.
