MECHANISM: Verified 12.5%-floor cosine decay

HYPOTHESIS: Raising the terminal learning-rate floor from 7.5% to 12.5% will exceed the current 9,233 correct predictions and recover the 9,240-result region.

INTENDED_EDIT: Change only the cosine schedule’s terminal floor, preserving the verified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.

EVIDENCE: Reference Design 2 used the otherwise identical implementation and achieved the strongest verified result—9,240 correct with 0.208897 cross-entropy—versus 9,233 correct for the current 7.5% floor; unsuccessful replications provide no contrary accuracy evidence.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE