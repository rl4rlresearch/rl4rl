MECHANISM: Best-batch learning-rate bracketing

HYPOTHESIS: At the proven 262,144-token batch, scaling all optimizer learning rates to 81.25% will retain at least 460M-token throughput and reduce val_bpb below 0.985487.

INTENDED_EDIT: Restore the best-performing 262,144-token single-microbatch configuration and reduce AdamW and Muon learning rates another 6.25% from the current best.

EVIDENCE: At 196,608 tokens, a 25% learning-rate reduction improved val_bpb from 0.986435 to 0.985713; at 262,144 tokens, a 12.5% reduction improved 0.985506 to 0.985487, motivating a measured step further in the same direction.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.4875   # 81.25% of the proven embedding LR
UNEMBEDDING_LR = 0.00325 # 81.25% of the proven lm_head LR
MATRIX_LR = 0.0325      # 81.25% of the proven Muon LR
SCALAR_LR = 0.40625     # 81.25% of the proven per-layer scalar LR
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 128  # one microbatch per optimizer step
>>>>>>> REPLACE