MECHANISM: Intermediate-batch learning-rate interpolation

HYPOTHESIS: A 229,376-token batch with optimizer learning rates scaled to 87.5% will retain at least 460M-token throughput and reduce val_bpb below 0.985506.

INTENDED_EDIT: Use a single 112-sequence microbatch per optimizer step and scale all AdamW and Muon learning rates linearly with the batch reduction.

EVIDENCE: Scaling the 196,608-token design’s learning rates to 75% improved val_bpb from 0.986435 to 0.985713, nearly matching the 262,144-token design’s 0.985506; testing their midpoint probes the remaining batch–update tradeoff while preserving cumulative learning-rate exposure.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens per optimizer step
EMBEDDING_LR = 0.525    # batch-linearly scaled token embedding LR
UNEMBEDDING_LR = 0.0035 # batch-linearly scaled lm_head LR
MATRIX_LR = 0.035       # batch-linearly scaled Muon LR
SCALAR_LR = 0.4375      # batch-linearly scaled per-layer scalar LR
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 112  # one microbatch per optimizer step
>>>>>>> REPLACE