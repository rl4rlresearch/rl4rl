MECHANISM: Bracketed 57.5% linear warmdown on the proven 256-token architecture

HYPOTHESIS: Starting linear warmdown at 42.5% of training will retain roughly 528M tokens while lowering val_bpb below 0.992358.

INTENDED_EDIT: Restore seven 256-token local-attention layers and set linear warmdown to 57.5% of the training window.

EVIDENCE: The 256-token architecture with 60% warmdown achieved the best val_bpb of 0.992358; 50% reached 0.992854 and 65% reached 0.992872, so 57.5% probes the more promising side of the bracketed optimum.

<<<<<<< SEARCH
        short_window = 7 * long_window // 64
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 224-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.575  # begin linear LR warmdown after 42.5% of the time budget
>>>>>>> REPLACE