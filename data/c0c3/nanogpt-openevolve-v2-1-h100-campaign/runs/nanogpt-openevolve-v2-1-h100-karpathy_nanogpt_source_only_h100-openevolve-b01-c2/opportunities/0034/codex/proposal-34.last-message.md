MECHANISM: Higher-frequency optimizer updates via quartered global batch

HYPOTHESIS: Reducing the optimizer batch to 131K tokens will approximately double update frequency again while retaining at least 480M training tokens, lowering val_bpb below 0.985657.

INTENDED_EDIT: Use one 64-sequence microbatch per optimizer step, preserve the proven 256-token local-attention architecture, and restore the best 60% linear warmdown.

EVIDENCE: Halving the batch from 524K to 262K tokens nearly doubled updates and improved val_bpb from 0.992358 to 0.985657 while still processing 518.8M tokens, strongly motivating another batch-size reduction.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
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
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens; one 64-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64   # efficient per-device batch size for 131K-token steps
>>>>>>> REPLACE