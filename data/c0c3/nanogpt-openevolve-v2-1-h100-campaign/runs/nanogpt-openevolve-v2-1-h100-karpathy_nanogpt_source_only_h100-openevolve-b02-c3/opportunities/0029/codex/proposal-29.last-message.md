MECHANISM: Middle-depth-concentrated ReLU² capacity allocation

HYPOTHESIS: Shifting upper-half MLP capacity from uniform 5.5× to 6× in layers 4–5 and 5× in layers 6–7 will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.

INTENDED_EDIT: Retain the best 2.25×/2.75× lower-half taper while redistributing unchanged upper-half capacity toward layers 4–5.

EVIDENCE: The current 2.25×/2.75×/5.5× design achieved the best val_bpb of 0.991682, while concentrating upper capacity in the final two layers with a 5×/6× split regressed to 0.992466; testing the reverse 6×/5× split directly probes whether capacity is more useful at middle depth without changing aggregate parameters or MLP compute.

<<<<<<< SEARCH
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
=======
        if layer_idx < config.n_layer // 4:
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        elif layer_idx < 3 * config.n_layer // 4:
            mlp_mult_quarters = 24  # 6x
        else:
            mlp_mult_quarters = 20  # 5x
>>>>>>> REPLACE