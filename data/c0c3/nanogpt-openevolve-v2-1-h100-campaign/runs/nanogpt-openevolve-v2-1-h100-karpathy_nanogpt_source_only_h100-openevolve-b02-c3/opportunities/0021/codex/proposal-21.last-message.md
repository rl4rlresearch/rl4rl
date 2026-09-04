MECHANISM: Stronger depth-tapered ReLU² capacity allocation

HYPOTHESIS: Moving from 3×/5× to 2.5×/5.5× lower/upper MLPs will preserve aggregate parameters and roughly 500M-token throughput while reducing val_bpb below 0.992431.

INTENDED_EDIT: Allocate 2.5× MLP width to the lower four layers and 5.5× width to the upper four, retaining the 4× average width and 50% linear warmdown.

EVIDENCE: Strengthening the taper from 3.5×/4.5× to 3×/5× improved val_bpb from 0.992809 to 0.992431 at nearly unchanged throughput, motivating one further tensor-core-aligned step along the same axis.

<<<<<<< SEARCH
        mlp_mult = 3 if layer_idx < config.n_layer // 2 else 5
        mlp_dim = mlp_mult * config.n_embd
=======
        mlp_mult_halves = 5 if layer_idx < config.n_layer // 2 else 11
        mlp_dim = mlp_mult_halves * config.n_embd // 2
>>>>>>> REPLACE