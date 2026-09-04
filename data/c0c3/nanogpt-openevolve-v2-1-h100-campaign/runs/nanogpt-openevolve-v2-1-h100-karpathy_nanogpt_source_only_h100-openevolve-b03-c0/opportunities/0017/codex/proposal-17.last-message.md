MECHANISM: Two-to-one grouped-query attention

HYPOTHESIS: Halving the KV-head count from four to two will reduce K/V projection, value-embedding, and optimizer costs enough to exceed 492.1M tokens while preserving all eight blocks and four query heads, achieving `val_bpb < 0.986491`.

INTENDED_EDIT: Configure two KV heads for the current four-query-head model, enabling grouped-query attention and proportionally smaller value embeddings.

EVIDENCE: Max-autotuning reached `val_bpb 0.986491` at 492.1M tokens but only 39.46% MFU; depth and MLP contraction hurt quality, so this targets redundant KV-side computation while retaining the validated depth, MLP capacity, query width, batching, and schedule.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE