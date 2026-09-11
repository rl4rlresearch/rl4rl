MECHANISM: Fixed-batch learning-rate continuation

HYPOTHESIS: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 2/3 of the 96-sequence baseline will achieve `val_bpb < 0.984431` by continuing the improvement observed as the scale decreased from 1.0 to 5/6 to 3/4.

INTENDED_EDIT: Reduce every optimizer learning rate from 3/4 to 2/3 of its 96-sequence baseline while preserving architecture, batching, schedules, and compilation.

EVIDENCE: At batch size 80, learning-rate scales of 1.0, 5/6, and 3/4 produced `val_bpb` values of `0.986929`, `0.985642`, and `0.984431`; the consistent improvement motivates one further comparable reduction to test and bracket the optimum.

<<<<<<< SEARCH
EMBEDDING_LR = 0.45     # 3/4 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.003  # 3/4 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.03        # 3/4 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.375       # 3/4 of the 96-sequence baseline for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.4              # 2/3 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.002666666667 # 2/3 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.026666666667      # 2/3 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.333333333333      # 2/3 of the 96-sequence baseline for per-layer scalars (Adam)
>>>>>>> REPLACE