MECHANISM: Quadratically interpolated fixed-batch learning-rate tuning

HYPOTHESIS: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 72.5% of the 96-sequence baseline will achieve `val_bpb < 0.984431`.

INTENDED_EDIT: Reduce embedding, unembedding, matrix, and scalar learning rates from 75% to 72.5% of their 96-sequence baselines while preserving all other settings.

EVIDENCE: At batch size 80, the 75% scale achieved `0.984431`, while 83.3% and 66.7% achieved `0.985642` and `0.984780`; quadratic interpolation of these bracketed results estimates the optimum near 72.7%.

<<<<<<< SEARCH
EMBEDDING_LR = 0.45     # 3/4 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.003  # 3/4 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.03        # 3/4 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.375       # 3/4 of the 96-sequence baseline for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.435    # 72.5% of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.0029 # 72.5% of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.029       # 72.5% of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.3625      # 72.5% of the 96-sequence baseline for per-layer scalars (Adam)
>>>>>>> REPLACE