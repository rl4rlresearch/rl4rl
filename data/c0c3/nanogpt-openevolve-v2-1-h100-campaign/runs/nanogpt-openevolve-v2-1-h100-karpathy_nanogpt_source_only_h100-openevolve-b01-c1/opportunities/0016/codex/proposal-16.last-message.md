MECHANISM: Finer-grained eight-head attention

HYPOTHESIS: Halving head dimension from 128 to 64 will increase the 512-wide model from four to eight full query/KV heads without reducing parameters or context, lowering val_bpb below 0.992286.

INTENDED_EDIT: Set the target attention head dimension to 64 while preserving model width, depth, value-embedding width, and the final full-context layer.

EVIDENCE: Two-to-one grouped-query attention reduced KV capacity and regressed val_bpb to 0.996550 despite higher throughput; repartitioning the unchanged full KV width into more heads tests richer attention structure without that capacity loss.

<<<<<<< SEARCH
HEAD_DIM = 128          # target head dimension for attention
=======
HEAD_DIM = 64           # finer-grained attention heads at unchanged model width
>>>>>>> REPLACE