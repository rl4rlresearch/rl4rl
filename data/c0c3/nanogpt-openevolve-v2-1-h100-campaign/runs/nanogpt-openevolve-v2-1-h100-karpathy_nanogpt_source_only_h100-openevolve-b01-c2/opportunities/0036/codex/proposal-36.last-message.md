MECHANISM: Bracketed optimizer-batch refinement at 229K tokens

HYPOTHESIS: A 229,376-token optimizer batch will preserve near-262K throughput while increasing update frequency, lowering val_bpb below 0.985657.

INTENDED_EDIT: Restore proven 256-token local attention and 60% linear warmdown, then use one 112-sequence microbatch per optimizer step.

EVIDENCE: The 262K batch achieved the best val_bpb of 0.985657, while 196K reached 0.986505 and 131K regressed further; 229K directly probes the narrow bracket between the two strongest batch sizes.

<<<<<<< SEARCH
        short_window = 3 * long_window // 32
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSS" # seven 192-token local layers, then one forced full-context anchor

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
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor

# Optimization
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens; one 112-sequence microbatch per step
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
DEVICE_BATCH_SIZE = 112  # one microbatch per 229,376-token optimizer step
>>>>>>> REPLACE