MECHANISM: 131K-token single-microbatch higher-frequency optimization

HYPOTHESIS: Halving the optimizer batch again on the proven 2.25×/2.75×/5.5× architecture will retain at least 440M training tokens while increasing parameter-update frequency enough to reduce val_bpb below 0.985237.

INTENDED_EDIT: Restore the best verified MLP taper, reduce the optimizer batch from 262K to 131K tokens, and use a 64-sequence device batch so each optimizer step remains one microbatch.

EVIDENCE: Reducing the optimizer batch from 524K to 262K tokens nearly doubled updates, retained 495.2M tokens, and improved val_bpb substantially from 0.991682 to 0.985237; another halving directly tests whether that strong update-frequency trend continues.

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
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
>>>>>>> REPLACE

<<<<<<< SEARCH
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)
=======
DEVICE_BATCH_SIZE = 64   # one 131K-token microbatch per optimizer step
>>>>>>> REPLACE