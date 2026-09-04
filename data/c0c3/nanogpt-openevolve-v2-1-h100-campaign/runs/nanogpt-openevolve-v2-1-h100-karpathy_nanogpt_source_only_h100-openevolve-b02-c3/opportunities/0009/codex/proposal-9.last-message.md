MECHANISM: Extended linear learning-rate annealing

HYPOTHESIS: Restoring full KV heads and beginning warmdown at 40% of the training window will preserve roughly 513M-token throughput while improving late-stage convergence below 0.993287 val_bpb.

INTENDED_EDIT: Restore the proven full multi-head 512-token SSSL architecture and increase LR warmdown from 50% to 60%.

EVIDENCE: Full KV heads with 50% warmdown achieved the best result, 0.993287 at 513.3M tokens; shortening warmdown to 40% slightly regressed to 0.993995 at similar throughput, motivating a test in the opposite direction.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
>>>>>>> REPLACE