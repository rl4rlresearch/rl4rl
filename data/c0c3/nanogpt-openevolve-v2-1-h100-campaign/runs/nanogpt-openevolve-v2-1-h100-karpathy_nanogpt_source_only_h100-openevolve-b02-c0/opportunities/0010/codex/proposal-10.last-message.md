MECHANISM: Square-root learning-rate scaling for the winning batch size

HYPOTHESIS: Scaling every optimizer learning rate by √0.5 will reduce gradient-noise-induced overstepping while preserving approximately 490M-token throughput, improving val_bpb below 0.987174.

INTENDED_EDIT: Reduce AdamW, Muon, and scalar learning rates by approximately 29% without changing the model, context length, or 262K-token batch.

EVIDENCE: Halving the batch from 524K to 262K tokens nearly preserved throughput, doubled updates, and improved val_bpb from 0.995558 to 0.987174; subsequent compute-shape changes failed, making learning-rate retuning at the winning configuration the clearest next test.

<<<<<<< SEARCH
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
EMBEDDING_LR = 0.424264 # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.002828 # learning rate for lm_head (Adam)
MATRIX_LR = 0.028284    # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.353553    # learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE