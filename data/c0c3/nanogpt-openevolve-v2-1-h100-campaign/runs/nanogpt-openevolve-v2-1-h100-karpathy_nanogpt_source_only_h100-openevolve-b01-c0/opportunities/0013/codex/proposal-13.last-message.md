MECHANISM: Batch-proportional learning-rate compensation

HYPOTHESIS: A 229,376-token batch with every optimizer learning rate scaled by 7/8 will beat 0.985318 val_bpb by retaining the prior 112-sequence run’s 13% higher update cadence while preventing excess cumulative parameter movement.

INTENDED_EDIT: Restore the nearly competitive 112-sequence single-microbatch configuration and linearly scale all AdamW and Muon learning rates with its batch-size reduction.

EVIDENCE: The unscaled 229,376-token run reached 0.985719 val_bpb on 501.6M tokens, only 0.000401 behind the best result while taking 2,187 rather than 1,933 optimizer steps; this motivates correcting its per-step learning rates rather than further refining batch size alone.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 7 * 2**15 # ~229K tokens per optimizer step
EMBEDDING_LR = 0.525    # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0035 # learning rate for lm_head (Adam)
MATRIX_LR = 0.035       # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4375      # learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 112  # per-device batch size (reduce if OOM)
>>>>>>> REPLACE