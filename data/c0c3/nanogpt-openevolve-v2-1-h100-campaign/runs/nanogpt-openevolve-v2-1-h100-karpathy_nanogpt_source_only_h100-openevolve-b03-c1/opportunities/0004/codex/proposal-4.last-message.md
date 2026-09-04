MECHANISM: Two-to-one grouped-query attention

HYPOTHESIS: Halving the KV-head count while retaining four query heads and the successful SSSL 512-token pattern will reduce projection, value-embedding, and attention bandwidth enough to beat val_bpb 0.993365 within five minutes without materially reducing attention quality.

INTENDED_EDIT: Use two KV heads for the four query heads, leaving model width, depth, batching, and attention windows unchanged.

EVIDENCE: The one-global-layer design processed more tokens but regressed to 0.994122, showing that removing global mixing sacrifices useful capacity; grouped-query attention instead preserves the winning two-global-layer topology while targeting redundant KV computation.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE