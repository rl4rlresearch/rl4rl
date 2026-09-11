MECHANISM: Two-to-one grouped-query attention

HYPOTHESIS: Halving KV heads from four to two will beat 0.993324 val_bpb by reducing projection, value-embedding, and optimizer work while preserving the proven 512-token SSSL attention layout and all four query heads.

INTENDED_EDIT: Configure two KV heads for the four-head model, enabling grouped-query attention without changing depth, model width, batch size, or attention windows.

EVIDENCE: The 512-token SSSL design achieved the best result at 0.993324 val_bpb; further window and global-layer reductions regressed, motivating a conservative reduction in KV width that leaves the successful context structure intact.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE