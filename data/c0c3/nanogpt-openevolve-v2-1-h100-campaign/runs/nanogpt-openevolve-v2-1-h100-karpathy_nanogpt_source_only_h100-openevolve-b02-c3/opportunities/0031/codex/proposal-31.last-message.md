MECHANISM: Refined lower-half feed-forward depth taper

HYPOTHESIS: A 2.125×/2.875× lower-half taper with uniform 5.5× upper MLPs will process roughly 500M tokens and reduce val_bpb below 0.991682.

INTENDED_EDIT: Interpolate between the best 2.25×/2.75× allocation and the competitive 2×/3× allocation while preserving total parameters, aggregate MLP compute, and all other settings.

EVIDENCE: The 2.25×/2.75× taper achieved the best val_bpb of 0.991682, narrowly outperforming 2×/3× at 0.991736; their tensor-core-aligned midpoint directly refines the strongest bracket.

<<<<<<< SEARCH
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 3   # 1.5x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 7   # 3.5x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_eighths = 17  # 2.125x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_eighths = 23  # 2.875x
        else:
            mlp_mult_eighths = 44  # 5.5x
        mlp_dim = mlp_mult_eighths * config.n_embd // 8
>>>>>>> REPLACE