MECHANISM: Locally refined linear cooldown duration

HYPOTHESIS: A 54% linear warmdown will beat the best observed val_bpb of 0.986676 by moving toward the local optimum implied by the 50%, 55%, and 60% results.

INTENDED_EDIT: Begin linear cooldown at 46% training progress instead of 40%, retaining all other model, optimizer, and batching settings.

EVIDENCE: Warmdown ratios of 50%, 55%, and 60% yielded val_bpb values of 0.98713, 0.986676, and 0.987466; the best result at 55% and steeper regression toward 60% motivate a slightly shorter 54% cooldown.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # longer linear refinement over the final 60% of training
=======
WARMDOWN_RATIO = 0.54   # locally refined linear cooldown over the final 54% of training
>>>>>>> REPLACE