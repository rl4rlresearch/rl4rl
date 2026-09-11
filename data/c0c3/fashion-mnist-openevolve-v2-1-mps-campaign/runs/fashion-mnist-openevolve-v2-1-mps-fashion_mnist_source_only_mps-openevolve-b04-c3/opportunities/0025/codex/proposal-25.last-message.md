MECHANISM: Conservative upper-side cosine-floor refinement

HYPOTHESIS: A 13.75% terminal learning-rate floor will exceed 9,240 correct predictions by extending the verified improvement from 10% to 12.5% while remaining closer to the current optimum than the unverified 15% setting and well below the harmful 20% floor.

INTENDED_EDIT: Raise only the cosine schedule’s terminal learning-rate floor from 12.5% to 13.75% of peak, preserving all other training and inference behavior.

EVIDENCE: The 12.5% floor achieved the best verified result at 9,240 correct, improving over 9,233 at 10%, while 20% fell to 9,226; the attempted 15% run timed out and supplied no validation evidence, making 13.75% the closest informative upper-side refinement.

<<<<<<< SEARCH
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.1375 + 0.8625 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE