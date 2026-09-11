MECHANISM: Further local-window compression with preserved global integration

HYPOTHESIS: Reducing the six local-attention layers from 512 to 256 tokens while retaining full-context layers 4 and 8 will lower val_bpb below 0.984125 by increasing token throughput without eliminating the proven global-context path.

INTENDED_EDIT: Restore the best verified 50% linear warmdown and change short attention windows to one-eighth of the 2048-token context.

EVIDENCE: Successive local-window reductions from 1024 to 768 and then 512 tokens improved val_bpb from 0.985746 to 0.984868 and 0.984125 while increasing processed tokens from 491.3M to 496.0M and 505.9M.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # 256-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # refine between the strongest 50% and 60% schedules
=======
WARMDOWN_RATIO = 0.5    # best verified linear warmdown
>>>>>>> REPLACE