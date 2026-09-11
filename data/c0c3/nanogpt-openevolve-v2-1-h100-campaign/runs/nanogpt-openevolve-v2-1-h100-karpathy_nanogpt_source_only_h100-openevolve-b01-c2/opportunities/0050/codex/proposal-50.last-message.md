MECHANISM: Half-step linear warmdown refinement

HYPOTHESIS: Using the proven 258,048-token batch with a 61.5% zero-ending linear warmdown will retain roughly 520M training tokens and lower `val_bpb` below 0.985181.

INTENDED_EDIT: Restore the best 126-sequence microbatch and move the linear decay start 1.5 percentage points earlier, halving the tested exposure reduction while preserving all peak learning rates and optimizer settings.

EVIDENCE: The 60% warmdown achieved 0.985181, while the symmetric exposure changes from a 63% warmdown and a 5% LR floor regressed modestly to 0.985414 and 0.985480; testing the midpoint toward the less harmful 63% result provides the tightest remaining schedule refinement.

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
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.615  # begin linear LR warmdown after 38.5% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 124  # one microbatch per 253,952-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE