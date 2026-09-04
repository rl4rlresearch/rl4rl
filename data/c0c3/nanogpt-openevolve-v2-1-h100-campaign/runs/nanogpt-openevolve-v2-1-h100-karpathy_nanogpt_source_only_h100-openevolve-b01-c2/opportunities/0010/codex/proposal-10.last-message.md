MECHANISM: Square-root batch-aware learning-rate scaling

HYPOTHESIS: Scaling all optimizer learning rates by \(1/\sqrt{2}\) for the halved 262K-token batch will retain its useful 1,560-update regime while reducing update noise and overshoot, lowering val_bpb below 0.995558.

INTENDED_EDIT: Keep the current SSSL architecture and single-microbatch training, but reduce Adam and Muon learning rates by \(1/\sqrt{2}\).

EVIDENCE: The current 262K-token design nearly matched the best result (0.996687 versus 0.995558) despite processing only 408.9M versus 497.0M tokens, but made 1,560 versus 948 optimizer updates using unchanged per-step learning rates; this motivates retuning update magnitude rather than abandoning the higher-frequency regime.

<<<<<<< SEARCH
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.424264 # sqrt-batch-scaled learning rate for token embeddings
UNEMBEDDING_LR = 0.002828 # sqrt-batch-scaled learning rate for lm_head
MATRIX_LR = 0.028284    # sqrt-batch-scaled learning rate for matrix parameters
SCALAR_LR = 0.353553    # sqrt-batch-scaled learning rate for per-layer scalars
>>>>>>> REPLACE