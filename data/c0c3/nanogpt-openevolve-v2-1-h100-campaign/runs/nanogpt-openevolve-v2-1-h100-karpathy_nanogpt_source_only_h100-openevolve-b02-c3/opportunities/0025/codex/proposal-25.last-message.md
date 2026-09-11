MECHANISM: Lower-half depth-localized feed-forward reallocation

HYPOTHESIS: Using 2× MLPs in layers 0–1 and 3× MLPs in layers 2–3, while retaining uniform 5.5× upper-layer MLPs, will preserve roughly 500M-token throughput and reduce val_bpb below 0.991816.

INTENDED_EDIT: Redistribute the lower half’s unchanged average 2.5× MLP capacity toward its later layers; leave the best-performing upper-half allocation and all other settings unchanged.

EVIDENCE: The 2.5×/5.5× design achieved the best val_bpb of 0.991816, while concentrating upper-half capacity into 5×/6× regressed to 0.992466; isolating redistribution within the lower half tests whether finer upward depth bias helps without disturbing the proven upper allocation.

<<<<<<< SEARCH
        mlp_mult_halves = 5 if layer_idx < config.n_layer // 2 else 11
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 4   # 2x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 6   # 3x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
>>>>>>> REPLACE