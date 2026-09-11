MECHANISM: Midpoint lower-half depth-localized feed-forward reallocation

HYPOTHESIS: Using 1.75× MLPs in layers 0–1, 3.25× in layers 2–3, and 5.5× in layers 4–7 will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

INTENDED_EDIT: Restore the proven uniform 5.5× upper-half allocation and interpolate the lower-half taper between the tested 2×/3× and 1.5×/3.5× designs while preserving total parameters and aggregate MLP compute.

EVIDENCE: The 2×/3× lower taper achieved the best observed 0.991736, while strengthening it to 1.5×/3.5× produced a near-tied 0.991793 at comparable throughput; their tensor-core-aligned midpoint most directly refines this narrowly bracketed optimum.

<<<<<<< SEARCH
        if layer_idx < config.n_layer // 2:
            mlp_mult_halves = 5   # 2.5x
        elif layer_idx < 3 * config.n_layer // 4:
            mlp_mult_halves = 10  # 5x
        else:
            mlp_mult_halves = 12  # 6x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 7   # 1.75x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 13  # 3.25x
        else:
            mlp_mult_quarters = 22  # 5.5x
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
>>>>>>> REPLACE