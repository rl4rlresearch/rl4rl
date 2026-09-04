MECHANISM: Shallower same-width transformer

HYPOTHESIS: Reducing depth from 8 to 7 retains the 512-dimensional width and four value-embedding tables while removing one transformer block, increasing training volume beyond 497M tokens enough to lower val_bpb below 0.995558.

INTENDED_EDIT: Set DEPTH to 7 without changing batching, learning rates, or the proven SSSL attention pattern.

EVIDENCE: The starting design achieved 0.995558 after 497.0M tokens, while every tested change that reduced token throughput regressed; a same-width one-layer reduction directly tests the opposite compute-quality tradeoff without altering attention kernels or optimizer behavior.

<<<<<<< SEARCH
DEPTH = 8               # number of transformer layers
=======
DEPTH = 7               # number of transformer layers
>>>>>>> REPLACE