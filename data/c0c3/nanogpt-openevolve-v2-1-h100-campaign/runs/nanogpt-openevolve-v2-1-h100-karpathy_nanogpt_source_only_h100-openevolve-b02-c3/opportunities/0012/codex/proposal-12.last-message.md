MECHANISM: Half-window cosine learning-rate annealing

HYPOTHESIS: Replacing the best 50% linear warmdown with an equal-duration cosine decay will preserve roughly 513M-token throughput while improving late-stage convergence enough to reduce val_bpb below 0.993287.

INTENDED_EDIT: Restore the proven 50% warmdown and change only its interpolation from linear to cosine, retaining the established architecture, batch size, optimizer, and zero final learning rate.

EVIDENCE: Linear warmdowns of 40%, 52%, and 60% all underperformed the 50% result of 0.993287 at similar throughput, indicating that 50% is the strongest duration and motivating a schedule-shape test at that fixed duration.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for cosine LR warmdown
>>>>>>> REPLACE

<<<<<<< SEARCH
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
=======
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        cosine = 0.5 * (1.0 - math.cos(math.pi * cooldown))
        return cosine + (1.0 - cosine) * FINAL_LR_FRAC
>>>>>>> REPLACE