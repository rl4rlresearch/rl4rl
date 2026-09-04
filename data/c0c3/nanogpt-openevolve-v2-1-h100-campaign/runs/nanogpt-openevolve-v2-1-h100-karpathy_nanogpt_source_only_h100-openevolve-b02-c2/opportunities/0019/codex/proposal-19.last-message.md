MECHANISM: Reverse-cosine learning-rate redistribution

HYPOTHESIS: A reverse-cosine 50% warmdown will lower val_bpb below 0.985746 by preserving the linear schedule’s total learning-rate budget while shifting updates opposite to the unsuccessful cosine schedule—smaller early-warmdown updates and larger late-warmdown refinement updates.

INTENDED_EDIT: Restore the best verified 50% warmdown start and replace linear interpolation with a monotonic reverse-cosine curve.

EVIDENCE: The 50% linear warmdown achieved 0.985746, while moving its start to 40%, 55%, or 60% did not improve it and a same-window cosine warmdown worsened val_bpb to 0.989121; this motivates holding the best window and testing the opposite within-window redistribution.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.55   # refine between the strongest 50% and 60% schedules
=======
WARMDOWN_RATIO = 0.5    # best verified warmdown window; test reverse-cosine redistribution
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine_cooldown = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        reverse_cosine = 2.0 * cooldown - cosine_cooldown
        return reverse_cosine + (1.0 - reverse_cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE