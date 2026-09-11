MECHANISM: Longer linear refinement tail

HYPOTHESIS: Extending linear warmdown from 50% to 60% of training will beat val_bpb 0.98713 by shifting learning-rate exposure from the plateau into later refinement.

INTENDED_EDIT: Restore the best design’s linear cooldown and begin it at 40% training progress instead of 50%.

EVIDENCE: The early-weighted cosine cooldown regressed to 0.990027, while the late-weighted schedule recovered to 0.987466; this indicates that preserving relatively larger late-training updates is preferable.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # longer linear refinement over the final 60% of training
>>>>>>> REPLACE

<<<<<<< SEARCH
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_cooldown = 0.5 - 0.5 * math.cos(math.pi * cooldown)
        return cosine_cooldown + (1 - cosine_cooldown) * FINAL_LR_FRAC
=======
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
>>>>>>> REPLACE