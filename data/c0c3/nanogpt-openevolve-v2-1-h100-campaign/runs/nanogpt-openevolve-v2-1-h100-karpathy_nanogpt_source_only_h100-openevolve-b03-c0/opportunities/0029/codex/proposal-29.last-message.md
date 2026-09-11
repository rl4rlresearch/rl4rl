MECHANISM: Upper-side fixed-batch learning-rate interpolation

HYPOTHESIS: At the unchanged 80-sequence batch, scaling all optimizer learning rates to 77.5% of their 96-sequence baselines will achieve `val_bpb < 0.984431`.

INTENDED_EDIT: Increase embedding, unembedding, matrix, and scalar learning rates from 75% to 77.5% of their established baselines while preserving every other setting.

EVIDENCE: The 75% scale is the observed optimum at `0.984431`; fitting the nearby 72.5%, 75%, and 83.3% results places a local minimum near 77%, making 77.5% the most informative untested upper-side interpolation.

<<<<<<< SEARCH
EMBEDDING_LR = 0.45     # 3/4 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.003  # 3/4 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.03        # 3/4 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.375       # 3/4 of the 96-sequence baseline for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.465    # 31/40 of the 96-sequence baseline for token embeddings (Adam)
UNEMBEDDING_LR = 0.0031 # 31/40 of the 96-sequence baseline for lm_head (Adam)
MATRIX_LR = 0.031       # 31/40 of the 96-sequence baseline for matrix parameters (Muon)
SCALAR_LR = 0.3875      # 31/40 of the 96-sequence baseline for per-layer scalars (Adam)
>>>>>>> REPLACE