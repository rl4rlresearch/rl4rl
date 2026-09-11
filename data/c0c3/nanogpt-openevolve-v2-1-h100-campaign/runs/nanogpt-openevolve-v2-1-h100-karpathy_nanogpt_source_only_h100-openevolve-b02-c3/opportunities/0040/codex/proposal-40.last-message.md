MECHANISM: Finer-grained eight-head context routing

HYPOTHESIS: On the proven 2.25×/2.75×/5.5× taper and 262K-token optimizer batch, replacing four 128-dimensional attention heads with eight 64-dimensional heads will retain at least 480M training tokens and reduce val_bpb below 0.985237 by enabling more specialized contextual routing at essentially unchanged projection and attention FLOPs.

INTENDED_EDIT: Restore the best verified MLP taper and optimizer batch, then challenge the shared assumption that four wide heads are the best context representation by doubling head count while preserving model width, KV width, parameter scale, sequence length, and attention-window pattern.

EVIDENCE: The 262K-token design achieved the best observed val_bpb of 0.985237 at 495.2M tokens. All available designs fixed HEAD_DIM=128, while broader windows increased contextual compute but regressed to 0.993720; finer head factorization tests richer learned context selection without paying for longer attention.

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
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
HEAD_DIM = 64           # eight finer-grained attention heads at 512 model width
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE