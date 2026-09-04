MECHANISM: Upper-bracket cosine-floor refinement

HYPOTHESIS: A 15% terminal learning-rate floor will exceed 9,240 correct predictions by extending the improvement from 10% to 12.5% while remaining below the harmful 20% floor.

INTENDED_EDIT: Raise only the cosine schedule’s terminal learning-rate floor from 7.5% to 15% of peak, preserving the qualified architecture, optimizer, augmentation, batch size, and logit-space flip ensemble.

EVIDENCE: The 12.5% floor achieved the best result at 9,240 correct, improving over 9,233 at 10%, whereas 20% fell to 9,226; 15% is the most informative untested point between the current optimum and the demonstrated upper failure region.

<<<<<<< SEARCH
    multiplier = 0.075 + 0.925 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE