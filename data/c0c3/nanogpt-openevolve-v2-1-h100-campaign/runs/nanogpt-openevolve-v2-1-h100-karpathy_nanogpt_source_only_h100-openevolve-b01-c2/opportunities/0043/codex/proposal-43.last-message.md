MECHANISM: Five-percent Muon trajectory-length increase at the best batch size

HYPOTHESIS: Using the proven 258,048-token batch with `MATRIX_LR=0.042` will retain roughly 522M training tokens and lower val_bpb below 0.985181.

INTENDED_EDIT: Restore the best 126-sequence microbatch and increase only the Muon matrix learning rate by 5%, retaining five polar iterations and all other settings.

EVIDENCE: The 258,048-token design achieved the best val_bpb of 0.985181 with 522.0M tokens; smaller batches showed that additional optimizer progress can help at comparable token counts, while four-step Muon regressed to 0.986319, motivating a conservative update-magnitude increase without weakening orthogonalization.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 15 * 2**14 # ~246K tokens; one 120-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
=======
TOTAL_BATCH_SIZE = 63 * 2**12 # ~258K tokens; one 126-sequence microbatch per step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.042       # 5% higher Muon step size at the best optimizer batch
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 120  # one microbatch per 245,760-token optimizer step
=======
DEVICE_BATCH_SIZE = 126  # one microbatch per 258,048-token optimizer step
>>>>>>> REPLACE