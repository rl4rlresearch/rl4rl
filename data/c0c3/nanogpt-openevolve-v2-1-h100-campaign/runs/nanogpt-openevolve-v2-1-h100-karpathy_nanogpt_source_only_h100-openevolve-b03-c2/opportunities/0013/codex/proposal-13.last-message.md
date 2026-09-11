MECHANISM: Interpolated linear refinement tail

HYPOTHESIS: A 55% linear warmdown will beat val_bpb 0.98713 because the 60% schedule nearly matched the 50% optimum, while shortening warmdown to 40% regressed substantially, suggesting the optimum lies slightly above 50%.

INTENDED_EDIT: Begin the unchanged linear cooldown at 45% training progress, midway between the best 50% and competitive 60% warmdown designs.

EVIDENCE: Warmdown ratios of 40%, 50%, and 60% produced val_bpb values of 0.989243, 0.98713, and 0.987466 respectively; the asymmetric degradation around 50% motivates testing a modest shift toward the stronger side.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.55   # linear refinement over the final 55% of training
>>>>>>> REPLACE