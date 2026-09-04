MECHANISM: Upward peak-learning-rate refinement at the best optimizer batch

HYPOTHESIS: A 212,992-token batch with peak learning rates scaled by 33/35 will process at least 490M tokens and reduce val_bpb below 0.983699.

INTENDED_EDIT: Restore the best 104-sequence single-microbatch configuration and raise every optimizer-group peak learning rate from 13/14 to 33/35 of its base value.

EVIDENCE: The 13/14 configuration achieved the best val_bpb of 0.983699, while lowering its learning-rate scale to 9/10 worsened val_bpb to 0.983937; a smaller upward step tests the remaining side of the local optimum without changing architecture, batch, or schedule.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens per optimizer step
EMBEDDING_LR = 0.6 * 36 / 35      # slightly refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 36 / 35  # slightly refined lm_head peak LR
MATRIX_LR = 0.04 * 36 / 35        # slightly refined Muon peak LR
SCALAR_LR = 0.5 * 36 / 35         # slightly refined scalar peak LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 112  # one 229K-token microbatch per optimizer step
=======
# Optimization
TOTAL_BATCH_SIZE = 13 * 2**14 # ~213K tokens per optimizer step
EMBEDDING_LR = 0.6 * 33 / 35      # upward-refined embedding peak LR
UNEMBEDDING_LR = 0.004 * 33 / 35  # upward-refined lm_head peak LR
MATRIX_LR = 0.04 * 33 / 35        # upward-refined Muon matrix peak LR
SCALAR_LR = 0.5 * 33 / 35         # upward-refined scalar peak LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 104  # one 213K-token microbatch per optimizer step
>>>>>>> REPLACE