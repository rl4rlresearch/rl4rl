MECHANISM: Cosine terminal annealing on the best depth-tapered ReLU² model

HYPOTHESIS: Restoring the verified 2.25×/2.75×/5.5× MLP taper and replacing its linear final-50% warmdown with an equal-duration cosine warmdown will process at least 500M tokens and reduce val_bpb below 0.991682.

INTENDED_EDIT: Restore the best verified MLP allocation, then use cosine rather than linear learning-rate decay during the existing 50% warmdown without changing peak or final learning rates.

EVIDENCE: The 2.25×/2.75×/5.5× taper achieved the best val_bpb, 0.991682, while its 2.125×/2.875× refinement regressed to 0.991814 and auxiliary prediction regressed to 1.000046; this motivates preserving the best architecture and testing the previously fixed annealing shape.

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
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine + (1.0 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE