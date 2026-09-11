MECHANISM: Bracketed peak-learning-rate refinement

HYPOTHESIS: A 229,376-token batch with all peak learning rates increased by 2.86% will process at least 490M tokens and reduce val_bpb below 0.984024.

INTENDED_EDIT: Restore the best 112-sequence single-microbatch design and scale every optimizer-group peak learning rate by 36/35.

EVIDENCE: The 229,376-token run achieved 0.984024; moving to a higher effective per-token rate at 196,608 tokens regressed to 0.985044, while scaling that run back by 6/7 recovered 0.984034. The opposite 262,144-token direction regressed further to 0.985237, so the asymmetric bracket motivates a small upward refinement near the proven optimum.

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
>>>>>>> REPLACE