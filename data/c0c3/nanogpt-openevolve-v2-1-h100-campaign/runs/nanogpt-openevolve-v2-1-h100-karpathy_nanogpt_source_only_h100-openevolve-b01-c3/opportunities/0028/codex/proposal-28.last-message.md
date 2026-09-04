MECHANISM: Quadratic learning-rate interpolation at the optimal batch size

HYPOTHESIS: At the proven 262,144-token batch, scaling all optimizer learning rates to 93.75% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.

INTENDED_EDIT: Restore the 262,144-token single-microbatch configuration and set every AdamW and Muon learning rate midway between the tested 87.5% and 100% settings.

EVIDENCE: At 262,144 tokens, 87.5% learning rates achieved 0.985487, 100% achieved 0.985506, and 81.25% worsened to 0.986294; these measurements bracket the apparent optimum near 93.75%.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
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
DEVICE_BATCH_SIZE = 64   # per-device batch size
=======
# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.5625   # 93.75% of the proven embedding LR
UNEMBEDDING_LR = 0.00375 # 93.75% of the proven lm_head LR
MATRIX_LR = 0.0375      # 93.75% of the proven Muon LR
SCALAR_LR = 0.46875     # 93.75% of the proven per-layer scalar LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE