MECHANISM: Best-batch learning-rate refinement

HYPOTHESIS: At the proven 262,144-token batch, scaling every optimizer learning rate to 87.5% will retain at least 460M-token throughput and reduce val_bpb below 0.985506.

INTENDED_EDIT: Restore the best-performing 262,144-token single-microbatch configuration and reduce all AdamW and Muon learning rates by 12.5%.

EVIDENCE: Scaling learning rates down at 196,608 tokens improved val_bpb from 0.986435 to 0.985713; testing the same moderate reduction while holding the best 262,144-token batch fixed isolates learning rate from the batch-size confound in the unsuccessful 229,376-token run.

<<<<<<< SEARCH
# Optimization
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
EMBEDDING_LR = 0.45     # batch-linearly scaled token embedding LR
UNEMBEDDING_LR = 0.003  # batch-linearly scaled lm_head LR
MATRIX_LR = 0.03        # batch-linearly scaled Muon LR
SCALAR_LR = 0.375       # batch-linearly scaled per-layer scalar LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
=======
# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.525    # 87.5% of the proven embedding LR
UNEMBEDDING_LR = 0.0035 # 87.5% of the proven lm_head LR
MATRIX_LR = 0.035       # 87.5% of the proven Muon LR
SCALAR_LR = 0.4375      # 87.5% of the proven per-layer scalar LR
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE