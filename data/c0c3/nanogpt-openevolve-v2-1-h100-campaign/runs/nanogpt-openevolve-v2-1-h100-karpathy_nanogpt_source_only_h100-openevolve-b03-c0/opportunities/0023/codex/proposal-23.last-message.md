MECHANISM: Batch-size-invariant learning-rate scaling

HYPOTHESIS: An 80-sequence batch with all learning rates scaled by 80/96 will achieve `val_bpb < 0.985780` by gaining lower-batch gradient noise without the 20% larger cumulative update budget that confounded the prior 80-sequence run.

INTENDED_EDIT: Reduce the one-microbatch batch size from 96 to 80 sequences and scale every optimizer learning rate by 5/6.

EVIDENCE: The prior unscaled 80-sequence run preserved nearly all token throughput but regressed to `0.986929`; because it performed 19% more updates at unchanged learning rates, it did not isolate batch-size effects from increased cumulative parameter movement.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 96 * 2048 # ~197K tokens per optimizer step; one 96-sequence microbatch
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 80 * 2048 # ~164K tokens per optimizer step; one 80-sequence microbatch
EMBEDDING_LR = 0.5      # 80/96-scaled learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0033333333333333335  # 80/96-scaled learning rate for lm_head (Adam)
MATRIX_LR = 0.03333333333333333         # 80/96-scaled learning rate for matrix parameters (Muon)
SCALAR_LR = 0.4166666666666667          # 80/96-scaled learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 96   # per-device batch size; one microbatch per optimizer step
=======
DEVICE_BATCH_SIZE = 80   # per-device batch size; one microbatch per optimizer step
>>>>>>> REPLACE