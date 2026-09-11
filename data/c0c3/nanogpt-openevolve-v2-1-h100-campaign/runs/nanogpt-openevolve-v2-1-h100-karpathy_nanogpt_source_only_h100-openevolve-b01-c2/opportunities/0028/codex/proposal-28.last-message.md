MECHANISM: Earlier linear warmdown on the proven 256-token attention baseline

HYPOTHESIS: Restoring seven 256-token local layers and starting linear warmdown at 40% of training will retain roughly 525M tokens while lowering val_bpb below 0.992854.

INTENDED_EDIT: Restore the best-performing 256-token local windows and extend linear warmdown from 50% to 60% of the training window.

EVIDENCE: The 256-token design achieved the best val_bpb, 0.992854; equal-duration cosine warmdown regressed to 0.995509, motivating a conservative refinement that preserves linear decay but reduces learning rates earlier.

<<<<<<< SEARCH
        short_window = 15 * long_window // 128
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven 240-token local layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 256-token local layers, then one forced full-context anchor
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
>>>>>>> REPLACE