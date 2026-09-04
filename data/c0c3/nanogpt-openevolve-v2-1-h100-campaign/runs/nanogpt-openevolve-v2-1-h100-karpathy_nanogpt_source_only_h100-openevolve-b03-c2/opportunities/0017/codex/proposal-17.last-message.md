MECHANISM: Nonzero terminal learning-rate floor

HYPOTHESIS: Retaining 5% of the initial learning rate at the end of the proven 55% cooldown will beat val_bpb 0.986663 by making the final updates effective without materially changing the successful early-training trajectory or throughput.

INTENDED_EDIT: Raise the linear cooldown endpoint from zero to 5% of each parameter group’s initial learning rate.

EVIDENCE: The late-weighted equal-area cooldown recovered to 0.987466 from the early-weighted cosine result of 0.990027 at comparable throughput, indicating that allocating more update magnitude to late refinement is beneficial; a small terminal floor tests that signal while preserving the best verified 55% cooldown duration.

<<<<<<< SEARCH
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
FINAL_LR_FRAC = 0.05    # retain small updates through the end of refinement
>>>>>>> REPLACE