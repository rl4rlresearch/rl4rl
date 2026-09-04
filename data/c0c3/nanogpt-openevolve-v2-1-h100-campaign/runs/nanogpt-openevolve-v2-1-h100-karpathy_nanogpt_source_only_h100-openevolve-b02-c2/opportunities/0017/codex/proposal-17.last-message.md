MECHANISM: Cosine-shaped terminal learning-rate annealing

HYPOTHESIS: Replacing the linear 50% warmdown with a cosine warmdown will lower val_bpb below 0.985746 by retaining larger updates early in the proven convergence window while tapering more gently toward zero near validation.

INTENDED_EDIT: Keep the best batch size, weight decay, and warmdown start fixed, changing only the interpolation within the warmdown from linear to cosine.

EVIDENCE: The 50% linear warmdown achieved the best val_bpb of 0.985746, while moving its start to 40%, 55%, or 60% did not improve it; this motivates preserving the best start time and testing schedule shape instead.

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown_progress = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 + math.cos(math.pi * cooldown_progress))
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * cosine
>>>>>>> REPLACE