MECHANISM: Aggressive depth-tapered ReLU² capacity allocation

HYPOTHESIS: Moving from 2.5×/5.5× to 2×/6× lower/upper MLPs will preserve aggregate parameters and at least 500M-token throughput while reducing val_bpb below 0.991816.

INTENDED_EDIT: Allocate 2× MLP width to the lower four layers and 6× width to the upper four, retaining the 4× average width and proven 50% linear warmdown.

EVIDENCE: Successively strengthening the taper from 3.5×/4.5× through 3×/5× to 2.5×/5.5× improved val_bpb from 0.992809 to 0.992431 to 0.991816 without reducing aggregate capacity or throughput, motivating one further tensor-core-aligned step.

<<<<<<< SEARCH
        mlp_mult_halves = 7 if layer_idx < config.n_layer // 2 else 9
=======
        mlp_mult_halves = 4 if layer_idx < config.n_layer // 2 else 12
>>>>>>> REPLACE