MECHANISM: Midpoint lower-half feed-forward taper

HYPOTHESIS: A 2.25×/2.75× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991736.

INTENDED_EDIT: Use 2.25× MLPs in layers 0–1, 2.75× in layers 2–3, and 5.5× in layers 4–7, preserving total parameters and aggregate MLP compute.

EVIDENCE: Uniform 2.5× lower MLPs achieved 0.991816, while shifting to 2×/3× improved to 0.991736 and stronger shifts regressed; this tests the unmeasured midpoint on the weaker-taper side of the apparent optimum.

<<<<<<< SEARCH
        mlp_mult_quarters = 9 if layer_idx < config.n_layer // 2 else 23
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
>>>>>>> REPLACE