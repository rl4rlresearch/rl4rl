MECHANISM: Intermediate-batch optimizer-frequency refinement

HYPOTHESIS: A 196,608-token optimizer batch will retain at least 500M training tokens while balancing the superior update frequency of the 262K-token design against the throughput loss of the 131K-token design, lowering val_bpb below 0.985657.

INTENDED_EDIT: Restore the proven 256-token local-attention windows and 60% linear warmdown, then use one 96-sequence microbatch per optimizer step.

EVIDENCE: Reducing batch size from 524K to 262K improved val_bpb from 0.992358 to 0.985657, but reducing it again to 131K regressed to 0.991781 while lowering processed tokens from 518.8M to 483.9M; 196K directly probes the bracketed batch-size optimum.

<<<<<<< SEARCH
        short_window = 7 * long_window // 64
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 224-token local layers, then one forced full-context anchor

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
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor

# Optimization
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens; one 96-sequence microbatch per step
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
DEVICE_BATCH_SIZE = 96   # one microbatch per 196,608-token optimizer step
>>>>>>> REPLACE