MECHANISM: Time-compressed learning-rate decay at the proven optimizer batch

HYPOTHESIS: Restoring the 524K-token batch and shortening warmdown from 50% to 40% will preserve roughly 513M-token throughput while increasing useful high-learning-rate training, reducing val_bpb below 0.993287.

INTENDED_EDIT: Restore two-microbatch gradient accumulation and delay linear LR decay until 60% of the training window.

EVIDENCE: The 512-token SSSL design with a 524K-token batch achieved the best result, 0.993287 val_bpb at 513.3M tokens; halving the batch reduced throughput to 383.5M tokens and regressed to 1.000127, so the proven batch should be restored while testing an untried schedule lever.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
>>>>>>> REPLACE