MECHANISM: Batch-scaled higher update density

HYPOTHESIS: A 64-sequence batch with learning rates scaled by 64/96 will achieve `val_bpb < 0.985642` by increasing optimization frequency and gradient stochasticity while preserving the cumulative learning-rate budget.

INTENDED_EDIT: Reduce the one-microbatch batch size from 80 to 64 sequences and scale every optimizer learning rate from the 96-sequence baseline by 2/3.

EVIDENCE: Scaling the 80-sequence learning rates by 80/96 improved `val_bpb` from `0.986929` to `0.985642` while retaining 489.7M tokens, showing that smaller batches benefit when their increased update count is compensated by proportional learning-rate scaling.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 80 * 2048 # ~164K tokens per optimizer step; one 80-sequence microbatch
EMBEDDING_LR = 0.5      # 80/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0033333333333333335  # 80/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.03333333333333333         # 80/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4166666666666667          # 80/96-scaled learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 64 * 2048 # ~131K tokens per optimizer step; one 64-sequence microbatch
EMBEDDING_LR = 0.4      # 64/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0026666666666666666  # 64/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.02666666666666667         # 64/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.3333333333333333          # 64/96-scaled learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 80   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 64   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE