MECHANISM: Single-microbatch intermediate batch sizing

HYPOTHESIS: Restoring the successful SSSL attention pattern and using a 393,216-token batch in one 192-sequence microbatch will achieve val_bpb below 0.995558 by increasing update density over the 524K baseline without incurring the severe per-token optimizer overhead observed at 262K.

INTENDED_EDIT: Restore SSSL attention and set both the global and device batch to 393,216 tokens, producing one optimizer update per microbatch.

EVIDENCE: The 524K SSSL design achieved 0.995558 with 948 updates, while 262K increased updates to 1500 but lost 104M tokens and worsened to 0.998629; meanwhile SSSS provided no meaningful throughput gain and worsened validation to 0.997009. An intermediate one-microbatch configuration directly tests the apparent update-density/throughput tradeoff.

<<<<<<< SEARCH
# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSS" # half context throughout, with the final layer forced to full context

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
WINDOW_PATTERN = "SSSL" # two full-context layers across the eight-layer stack

# Optimization
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens per optimizer step
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
DEVICE_BATCH_SIZE = 192  # one 393K-token microbatch per optimizer step
>>>>>>> REPLACE