MECHANISM: Quadratically interpolated peak-learning-rate refinement

HYPOTHESIS: A 212,992-token batch with peak learning rates scaled by 23/25 will process at least 490M tokens and reduce val_bpb below 0.983699.

INTENDED_EDIT: Use a 104-sequence single microbatch and scale every optimizer-group peak learning rate to 23/25 of its base value.

EVIDENCE: At the fixed 212,992-token batch, scales of 9/10, 13/14, and 33/35 produced val_bpb of 0.983937, 0.983699, and 0.984113; quadratic interpolation places the estimated optimum near 0.919, motivating the nearby exact scale 23/25.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 96   # one 197K-token microbatch per optimizer step
=======
# Optimization
TOTAL_BATCH_SIZE = 13 * 2**14 # ~213K tokens per optimizer step
EMBEDDING_LR = 0.6 * 23 / 25      # interpolated embedding peak LR
UNEMBEDDING_LR = 0.004 * 23 / 25  # interpolated lm_head peak LR
MATRIX_LR = 0.04 * 23 / 25        # interpolated Muon matrix peak LR
SCALAR_LR = 0.5 * 23 / 25         # interpolated scalar peak LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 104  # one 213K-token microbatch per optimizer step
>>>>>>> REPLACE