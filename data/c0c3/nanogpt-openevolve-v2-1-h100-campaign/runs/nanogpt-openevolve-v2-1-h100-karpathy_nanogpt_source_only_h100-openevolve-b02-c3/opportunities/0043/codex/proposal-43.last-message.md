MECHANISM: Token-normalized smaller-batch learning rates

HYPOTHESIS: A 196,608-token batch with all peak learning rates scaled by 6/7 will process at least 490M tokens and reduce val_bpb below 0.984024 by preserving the best run’s cumulative optimizer displacement while benefiting from more frequent, less correlated updates.

INTENDED_EDIT: Restore four 128-dimensional heads, use a 96-sequence single microbatch, and scale every optimizer-group learning rate by the 196,608/229,376 batch ratio.

EVIDENCE: The 229,376-token batch achieved 0.984024, while 196,608 tokens achieved a close 0.985044 and 493.9M tokens using unchanged learning rates but 16% more updates; scaling rates by 6/7 isolates whether excessive cumulative update magnitude caused that regression.

<<<<<<< SEARCH
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 64           # eight finer-grained attention heads at 512 model width
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
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
HEAD_DIM = 128          # four 128-dimensional attention heads at 512 model width
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

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
>>>>>>> REPLACE