MECHANISM: Single-microbatch, higher-update-frequency training

HYPOTHESIS: Using spare H100 memory to eliminate gradient accumulation while reducing the global batch by 25% will preserve token throughput, increase optimizer updates by roughly 33%, and lower val_bpb below 0.995558.

INTENDED_EDIT: Raise the device batch to 192, set the global batch to 393,216 tokens, scale learning rates by 0.75 to preserve the token-normalized update budget, and extend the Muon momentum ramp to the same token horizon.

EVIDENCE: The starting design reaches val_bpb 0.995558 with 948 steps and only 45,060 MB peak VRAM; this leaves enough H100 memory to trade two 128-sequence microbatches for one 192-sequence microbatch and obtain more optimization steps.

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
=======
TOTAL_BATCH_SIZE = 3 * 2**17 # ~393K tokens; one 192-sequence microbatch
EMBEDDING_LR = 0.45     # 0.75x LR preserves update budget with 0.75x batch
UNEMBEDDING_LR = 0.003  # learning rate for lm_head (Adam)
MATRIX_LR = 0.03        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.375       # learning rate for per-layer scalars (Adam)
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 192  # consume spare H100 memory and eliminate accumulation
>>>>>>> REPLACE

<<<<<<< SEARCH
def get_muon_momentum(step):
    frac = min(step / 300, 1)
=======
def get_muon_momentum(step):
    frac = min(step / 400, 1)
>>>>>>> REPLACE