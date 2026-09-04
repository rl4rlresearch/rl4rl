MECHANISM: Shallower local-attention backbone

HYPOTHESIS: Removing one 256-token local block while preserving model width, four value embeddings, and the final full-context layer will increase throughput enough to lower val_bpb below 0.992286.

INTENDED_EDIT: Reduce transformer depth from eight to seven layers; rounding keeps n_embd at 512 and the last layer remains full-context.

EVIDENCE: The current 8-layer design reached 0.992286 after throughput gains from removing an earlier global-attention layer, while removing the final global layer sharply regressed to 1.015479. This motivates reducing cheaper local depth while retaining the empirically essential final global layer.

<<<<<<< SEARCH
DEPTH = 8               # number of transformer layers
=======
DEPTH = 7               # number of transformer layers
>>>>>>> REPLACE