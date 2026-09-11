MECHANISM: Finer-grained attention-head factorization

HYPOTHESIS: Splitting the unchanged 512-dimensional attention space into eight 64-dimensional heads will reduce val_bpb below 0.984227 without materially reducing throughput or changing parameter count.

INTENDED_EDIT: Change HEAD_DIM from 128 to 64, increasing query and key/value heads from four to eight while preserving model width, depth, context windows, optimizer, and schedule.

EVIDENCE: Adding a third global-attention layer improved val_bpb from 0.984312 to 0.984227 despite processing fewer tokens, indicating attention representation capacity is limiting; finer head partitioning tests additional relational subspaces without adding attention FLOPs or parameters.

<<<<<<< SEARCH
HEAD_DIM = 128          # target head dimension for attention
=======
HEAD_DIM = 64           # finer-grained attention: 8 heads at unchanged 512 model dimension
>>>>>>> REPLACE