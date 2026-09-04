MECHANISM: Refined weak-side lower-half feed-forward taper

HYPOTHESIS: A 2.375×/2.625× lower-half taper with uniform 5.5× upper MLPs will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.

INTENDED_EDIT: Restore the best architecture’s aggregate MLP allocation while testing the tensor-core-aligned midpoint between its 2.25×/2.75× lower taper and the competitive uniform 2.5× lower allocation.

EVIDENCE: The 2.25×/2.75× taper achieved the best observed 0.991682, outperforming uniform 2.5× at 0.991816; the previously tested stronger-side midpoint regressed, so this tests the unmeasured weaker-side midpoint.

<<<<<<< SEARCH
        if layer_idx < config.n_layer // 4:
            mlp_mult_halves = 4   # 2x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_halves = 6   # 3x
        else:
            mlp_mult_halves = 11  # 5.5x
        mlp_dim = mlp_mult_halves * config.n_embd // 2
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_eighths = 19  # 2.375x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_eighths = 21  # 2.625x
        else:
            mlp_mult_eighths = 44  # 5.5x
        mlp_dim = mlp_mult_eighths * config.n_embd // 8
>>>>>>> REPLACE