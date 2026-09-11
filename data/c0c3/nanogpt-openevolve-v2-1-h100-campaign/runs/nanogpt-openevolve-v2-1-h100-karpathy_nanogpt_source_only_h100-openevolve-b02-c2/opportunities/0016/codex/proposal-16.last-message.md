MECHANISM: Bracketed learning-rate warmdown refinement

HYPOTHESIS: A 55% LR warmdown in the proven 262K-token, weight-decay-0.1 configuration will lower val_bpb below 0.985746 by beginning convergence slightly earlier than the best 50% schedule without the excess early decay observed at 60%.

INTENDED_EDIT: Restore the best 262K-token single-microbatch configuration and normalized Muon weight decay, then set WARMDOWN_RATIO to 0.55.

EVIDENCE: At weight decay 0.1, 50% warmdown achieved the best val_bpb of 0.985746, while 40% worsened to 0.987004 and 60% narrowly worsened to 0.985949; the asymmetric bracket motivates testing the midpoint between the two strongest schedules.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; one device batch
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.1      # normalize cumulative cautious decay for the higher update cadence
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.55   # refine between the strongest 50% and 60% schedules
>>>>>>> REPLACE