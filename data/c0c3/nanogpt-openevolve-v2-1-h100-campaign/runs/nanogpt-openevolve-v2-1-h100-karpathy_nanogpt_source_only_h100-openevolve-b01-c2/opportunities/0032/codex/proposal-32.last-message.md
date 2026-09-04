MECHANISM: Mid-stack global-attention relay

HYPOTHESIS: On the proven 256-token, 60%-warmdown baseline, replacing the fourth local layer with full-context attention will retain at least 490M tokens while lowering val_bpb below 0.992358.

INTENDED_EDIT: Restore 256-token local windows and 60% linear warmdown, then use full-context attention at layers four and eight.

EVIDENCE: The 256-token baseline with 60% warmdown achieved the best val_bpb of 0.992358, while removing the final global anchor worsened val_bpb to 1.001821; this tests whether an earlier global mixing stage adds similar context value.

<<<<<<< SEARCH
        short_window = 3 * long_window // 32
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 192-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSL" # six 256-token local layers with full-context layers 4 and 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
>>>>>>> REPLACE