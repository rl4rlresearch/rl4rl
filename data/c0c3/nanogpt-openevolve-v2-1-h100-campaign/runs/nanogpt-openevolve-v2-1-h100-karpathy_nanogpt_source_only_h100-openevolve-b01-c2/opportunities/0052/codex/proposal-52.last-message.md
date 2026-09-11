MECHANISM: Later-start zero-ending linear warmdown continuation

HYPOTHESIS: A 57% zero-ending linear warmdown with the proven 258,048-token batch will retain roughly 520M training tokens and lower `val_bpb` below 0.985148.

INTENDED_EDIT: Restore the best 126-sequence microbatch and move linear-decay onset 1.5 percentage points later than the best 58.5% schedule, preserving all other hyperparameters.

EVIDENCE: At 258,048 tokens per step, progressively shortening warmdown from 63% to 61.5%, 60%, and 58.5% improved `val_bpb` from 0.985414 to 0.985321, 0.985181, and 0.985148; 57% is the next symmetric continuation and tests whether that trend persists.

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
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.57   # begin linear LR warmdown after 43% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE