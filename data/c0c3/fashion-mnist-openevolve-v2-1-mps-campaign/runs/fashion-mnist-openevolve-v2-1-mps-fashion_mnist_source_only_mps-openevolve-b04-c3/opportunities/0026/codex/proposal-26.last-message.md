MECHANISM: Verified 12.5%-floor cosine optimization

HYPOTHESIS: Raising the terminal learning-rate floor from 7.5% to 12.5% will exceed 9,233 correct predictions and reproduce the demonstrated 9,240-result region.

INTENDED_EDIT: Change only the cosine schedule’s terminal floor to 12.5% of peak, preserving the architecture, augmentation, optimizer, batch size, and logit-space flip ensemble.

EVIDENCE: Reference Design 2 used this exact schedule and achieved the best verified result—9,240 correct—outperforming the current 7.5% floor’s 9,233 and the 10% floor’s 9,233.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE