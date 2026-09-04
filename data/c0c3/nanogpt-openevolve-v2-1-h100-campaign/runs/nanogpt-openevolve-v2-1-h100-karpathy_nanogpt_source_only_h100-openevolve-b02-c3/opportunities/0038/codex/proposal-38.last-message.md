MECHANISM: Single-microbatch higher-frequency optimization

HYPOTHESIS: Halving the optimizer batch to 262K tokens on the best 2.25×/2.75×/5.5× architecture will provide roughly twice as many parameter updates, retain at least 450M training tokens, and reduce val_bpb below 0.991682.

INTENDED_EDIT: Restore the best verified MLP taper and reduce gradient accumulation from two microbatches to one while preserving all learning rates and schedules.

EVIDENCE: The 2.25×/2.75×/5.5× design achieved the best val_bpb, 0.991682; subsequent width, schedule, auxiliary-loss, and attention changes regressed, while every verified design retained the same 524K-token optimizer batch, making update frequency an untested orthogonal mechanism.

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
            mlp_mult_quarters = 9   # 2.25x
        elif layer_idx < config.n_layer // 2:
            mlp_mult_quarters = 11  # 2.75x
        else:
            mlp_mult_quarters = 22  # 5.5x
        mlp_dim = mlp_mult_quarters * config.n_embd // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE