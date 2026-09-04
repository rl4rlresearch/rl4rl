MECHANISM: Quadratically interpolated batch-size and learning-rate scaling

HYPOTHESIS: An 84-sequence batch with learning rates scaled by 84/96 will achieve `val_bpb < 0.985642` by operating near the batch-size optimum implied by the 64-, 80-, and 96-sequence results while preserving cumulative parameter movement.

INTENDED_EDIT: Increase the one-microbatch batch size from 80 to 84 sequences and scale every optimizer learning rate from the 96-sequence baseline by 7/8.

EVIDENCE: Batch-scaled runs achieved `val_bpb` values of `0.985916`, `0.985642`, and `0.985780` at batch sizes 64, 80, and 96 respectively; quadratic interpolation places the estimated local optimum near 83 sequences.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 80 * 2048 # ~164K tokens per optimizer step; one 80-sequence microbatch
EMBEDDING_LR = 0.5      # 80/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0033333333333333335  # 80/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.03333333333333333         # 80/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4166666666666667          # 80/96-scaled learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 84 * 2048 # ~172K tokens per optimizer step; one 84-sequence microbatch
EMBEDDING_LR = 0.525     # 84/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0035  # 84/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.035        # 84/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4375       # 84/96-scaled learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 80   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 84   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE