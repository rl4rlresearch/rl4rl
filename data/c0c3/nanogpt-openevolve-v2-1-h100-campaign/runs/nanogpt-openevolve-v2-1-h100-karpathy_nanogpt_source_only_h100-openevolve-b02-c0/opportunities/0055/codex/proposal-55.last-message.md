MECHANISM: Stronger post-global attention-window compression

HYPOTHESIS: Reducing post-global local-attention windows from 320 to 256 tokens will lower val_bpb below 0.983497 by increasing training throughput while the depth-4 global layer already supplies long-range context.

INTENDED_EDIT: Keep the first three local layers at 424 tokens and both global layers unchanged, while shortening the three local layers after the first global layer to 256 tokens.

EVIDENCE: Compressing only post-global windows from 424 to 320 improved val_bpb from 0.983635 to 0.983497 and increased tokens from 506.2M to 510.1M, whereas compressing early layers instead regressed to 0.984264; this motivates testing whether later local context can be reduced further.

<<<<<<< SEARCH
        post_global_window = 5 * long_window // 32
=======
        post_global_window = long_window // 8
>>>>>>> REPLACE