MECHANISM: Verified 12.5%-floor cosine optimization

HYPOTHESIS: Raising the terminal learning-rate floor from 7.5% to 12.5% will improve validation correctness from 9,233 toward the verified 9,240-result region.

INTENDED_EDIT: Change only the cosine schedule’s terminal floor, preserving the architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.

EVIDENCE: Reference Design 2 achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—using the identical implementation with a 12.5% floor, versus 9,233 correct for the current 7.5% floor.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE