MECHANISM: Upper-side Muon learning-rate midpoint refinement

HYPOTHESIS: Using `MATRIX_LR=0.041` with the proven 258,048-token batch will retain roughly 520M training tokens and lower val_bpb below 0.985181.

INTENDED_EDIT: Restore the best 126-sequence microbatch and increase only the Muon matrix learning rate from 0.040 to 0.041.

EVIDENCE: The 258,048-token batch with `MATRIX_LR=0.040` achieved the best val_bpb of 0.985181, while 0.042 regressed to 0.985838; 0.041 tests the unresolved midpoint of that upper learning-rate bracket.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 31 * 2**13 # ~254K tokens; one 124-sequence microbatch per step
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
DEVICE_BATCH_SIZE = 124  # one microbatch per 253,952-token optimizer step
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.041       # upper midpoint between the best 0.040 and tested 0.042
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE