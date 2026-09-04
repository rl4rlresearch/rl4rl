MECHANISM: Power-of-two microbatch scaling with reduced gradient noise

HYPOTHESIS: A 262,144-token optimizer batch with the best 58.5% zero-ending linear warmdown will retain roughly 520M training tokens and lower `val_bpb` below 0.985148.

INTENDED_EDIT: Increase the single-step microbatch from 120 to 128 sequences and adopt the best verified 58.5% warmdown, preserving the architecture and optimizer learning rates.

EVIDENCE: Increasing the batch from 245,760 to 253,952 and then 258,048 tokens improved `val_bpb` from 0.985506 to 0.985215 and 0.985181; combining the next power-of-two microbatch with the best verified 58.5% schedule tests whether that trend continues.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 15 * 2**14 # ~246K tokens; one 120-sequence microbatch per step
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
DEVICE_BATCH_SIZE = 120  # one microbatch per 245,760-token optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # 262K tokens; one 128-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.585  # begin linear LR warmdown after 41.5% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # one microbatch per 262,144-token optimizer step
>>>>>>> REPLACE