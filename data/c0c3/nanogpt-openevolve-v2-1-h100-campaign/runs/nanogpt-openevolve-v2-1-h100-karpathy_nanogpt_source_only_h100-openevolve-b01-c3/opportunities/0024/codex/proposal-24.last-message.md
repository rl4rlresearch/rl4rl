MECHANISM: Batch-linear learning-rate normalization

HYPOTHESIS: A 196,608-token batch with every optimizer learning rate scaled by 0.75 will preserve at least 450M-token throughput while matching the successful 262,144-token run’s cumulative update magnitude, reducing val_bpb below 0.985506.

INTENDED_EDIT: Restore uniform 4.375× MLPs, use the proven 196,608-token single-microbatch configuration, and scale all AdamW and Muon learning rates in proportion to batch size.

EVIDENCE: The 196,608-token design reached 0.986435 versus 0.985506 at 262,144 tokens, but used the same learning rates across 32% more updates; scaling rates by 0.75 preserves its finer update frequency without increasing cumulative learning-rate exposure.

<<<<<<< SEARCH
        expansion_quarters = 18 if layer_idx < config.n_layer // 2 else 17
        hidden_dim = expansion_quarters * config.n_embd // 4
=======
        hidden_dim = 35 * config.n_embd // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 3 * 2**16 # ~197K tokens per optimizer step
EMBEDDING_LR = 0.45     # batch-linearly scaled token embedding LR
UNEMBEDDING_LR = 0.003  # batch-linearly scaled lm_head LR
MATRIX_LR = 0.03        # batch-linearly scaled Muon LR
SCALAR_LR = 0.375       # batch-linearly scaled per-layer scalar LR
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 96   # one microbatch per optimizer step
>>>>>>> REPLACE