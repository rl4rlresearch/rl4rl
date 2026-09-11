MECHANISM: Verified nonzero-floor cosine optimization

HYPOTHESIS: Raising the terminal learning-rate floor from 7.5% to 12.5% will improve correctness from 9,233 toward the verified 9,240-result region.

INTENDED_EDIT: Change only the cosine schedule’s terminal floor, preserving the verified architecture, augmentation, optimizer, batch size, and logit-space flip ensemble.

EVIDENCE: Reference Design 2 differs from the current implementation only in this floor and achieved 9,240 correct with 0.208897 cross-entropy, versus 9,233 correct for the current 7.5% floor; failed replications produced no contrary accuracy evidence.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE