MECHANISM: Two-to-one grouped-query attention at the proven local-attention geometry

HYPOTHESIS: Halving KV heads while restoring 512-token SSSL attention will preserve modeling quality, process more than 513.3M tokens, and reduce val_bpb below 0.993287.

INTENDED_EDIT: Restore quarter-context short windows and use two KV heads shared across four query heads.

EVIDENCE: The 512-token SSSL design achieved the best val_bpb, 0.993287 at 513.3M tokens, while changing its windows, global-layer pattern, or optimizer batch regressed; reducing redundant KV projections tests an orthogonal efficiency lever without disturbing the proven receptive field or optimizer batch.

<<<<<<< SEARCH
        short_window = long_window // 8
=======
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // 2), n_embd=model_dim,
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=eighth context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context
>>>>>>> REPLACE