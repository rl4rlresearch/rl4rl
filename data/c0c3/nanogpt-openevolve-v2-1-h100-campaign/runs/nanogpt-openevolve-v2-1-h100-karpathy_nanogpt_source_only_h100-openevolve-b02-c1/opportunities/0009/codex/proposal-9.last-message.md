MECHANISM: Two-to-one grouped-query attention

HYPOTHESIS: Sharing each key/value head across two of the four query heads will reduce projection, value-embedding, and optimizer-update costs enough to exceed the baseline’s 497M tokens while preserving query-head geometry, lowering val_bpb below 0.995558.

INTENDED_EDIT: Set `n_kv_head` to half of `n_head`, producing four query heads and two key/value heads.

EVIDENCE: The four-head baseline achieved 0.995558 at 497M tokens, whereas eight smaller heads reduced throughput to 472.9M and regressed to 1.0038; grouped-query attention retains the successful four-query-head layout while targeting K/V-side compute and bandwidth instead.

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE