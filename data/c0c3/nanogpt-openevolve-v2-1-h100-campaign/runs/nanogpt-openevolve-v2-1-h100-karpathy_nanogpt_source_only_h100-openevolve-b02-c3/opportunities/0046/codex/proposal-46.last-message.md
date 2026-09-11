MECHANISM: Downward peak-learning-rate refinement at the best optimizer batch

HYPOTHESIS: A 212,992-token batch with all peak learning rates reduced from 13/14 to 9/10 of their base values will process at least 490M tokens and reduce val_bpb below 0.983699.

INTENDED_EDIT: Restore the best 104-sequence single-microbatch configuration and lower every optimizer-group peak learning rate by 3.08%.

EVIDENCE: The 212,992-token design achieved the best val_bpb, 0.983699. At fixed 196,608- and 229,376-token batches, increasing peak learning rates worsened val_bpb by 0.001010 and 0.000689 respectively, motivating a conservative downward refinement while holding the best batch fixed.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
EMBEDDING_LR = 0.6 * 6 / 7      # token-normalized learning rate for embeddings
UNEMBEDDING_LR = 0.004 * 6 / 7  # token-normalized learning rate for lm_head
MATRIX_LR = 0.04 * 6 / 7        # token-normalized learning rate for Muon matrices
SCALAR_LR = 0.5 * 6 / 7         # token-normalized learning rate for scalars
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
EMBEDDING_LR = 0.6 * 9 / 10      # downward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 9 / 10  # downward-refined lm_head peak LR
MATRIX_LR = 0.04 * 9 / 10        # downward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 9 / 10         # downward-refined scalar peak LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 104  # one 213K-token microbatch per optimizer step
>>>>>>> REPLACE