MECHANISM: Shorter linear cooldown with greater late-training learning-rate exposure

HYPOTHESIS: On the proven 128-sequence, 262K-token update geometry, shortening linear warmdown from 50% to 40% will beat val_bpb 0.98713 by retaining larger learning rates later in training.

INTENDED_EDIT: Restore Reference Design 3’s efficient single-microbatch geometry and begin linear cooldown at 60% training progress.

EVIDENCE: Reference Design 3 achieved 0.98713 at 489.7M tokens; extending warmdown to 60% produced 0.987466, while the early-weighted cosine schedule regressed to 0.990027, motivating the opposite adjustment toward greater late-training learning-rate exposure.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
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
DEVICE_BATCH_SIZE = 96   # two microbatches per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.4    # linear refinement over the final 40% of training
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE