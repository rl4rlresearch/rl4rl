MECHANISM: Single terminal global-attention layer

HYPOTHESIS: On the best 2.25×/2.75×/5.5× MLP taper, removing the redundant layer-3 full-context attention pass will increase throughput beyond 510.7M tokens while the forced full-context final layer preserves global mixing, reducing val_bpb below 0.991682.

INTENDED_EDIT: Restore the best verified MLP allocation and change the attention pattern from SSSL to SSSS, leaving only the automatically forced final layer at full context.

EVIDENCE: The 2.25×/2.75×/5.5× design achieved the best verified val_bpb of 0.991682 at 510.7M tokens; subsequent MLP refinements and schedule changes did not improve it, while every verified design retained two full-context layers, making removal of the earlier full-context pass an untested throughput–context tradeoff.

<<<<<<< SEARCH
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 7   # 1.75x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 13  # 3.25x
        else:
            mlp_mult_quarters = 22  # 5.5x
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context
=======
WINDOW_PATTERN = "SSSS" # quarter-context attention except forced full-context final layer
>>>>>>> REPLACE