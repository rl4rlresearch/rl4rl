MECHANISM: Stronger lower-half depth-localized feed-forward reallocation

HYPOTHESIS: Shifting the lower-half allocation from 2×/3× to 1.5×/3.5× while retaining uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

INTENDED_EDIT: Use 1.5× MLPs in layers 0–1, 3.5× in layers 2–3, and 5.5× in layers 4–7, preserving total parameters, aggregate MLP compute, and all other settings.

EVIDENCE: Redistributing the best 2.5× lower-half capacity to 2×/3× produced the best observed val_bpb of 0.991736 at 510.1M tokens, while redistributing upper-layer capacity regressed; this tests one further lower-only shift without disturbing the proven upper allocation.

<<<<<<< SEARCH
        mlp_mult_halves = 4 if layer_idx < config.n_layer // 2 else 12
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 3   # 1.5x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 7   # 3.5x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
>>>>>>> REPLACE