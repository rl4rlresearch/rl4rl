MECHANISM: Fixed-batch learning-rate continuation

HYPOTHESIS: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 75% of the 96-sequence baseline will achieve `val_bpb < 0.985642` by continuing the improvement observed when the scale was reduced from 1.0 to 5/6.

INTENDED_EDIT: Keep architecture, batching, compilation, and schedules fixed while reducing every optimizer learning rate by 10% from the current values.

EVIDENCE: At batch size 80, reducing the learning-rate scale from 1.0 to 5/6 improved `val_bpb` from `0.986929` to `0.985642` with nearly unchanged token throughput; testing 3/4 extends that controlled trend and brackets whether the optimum lies below 5/6.

<<<<<<< SEARCH
EMBEDDING_LR = 0.5      # 80/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0033333333333333335  # 80/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.03333333333333333         # 80/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4166666666666667          # 80/96-scaled learning rate for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.45     # 3/4 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.003  # 3/4 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.03        # 3/4 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.375       # 3/4 of the 96-sequence baseline for per-layer scalars (Adam)
>>>>>>> REPLACE