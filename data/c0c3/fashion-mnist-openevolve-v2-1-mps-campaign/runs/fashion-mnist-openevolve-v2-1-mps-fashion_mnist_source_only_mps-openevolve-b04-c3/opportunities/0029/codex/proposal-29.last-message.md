MECHANISM: Fine-grained cosine-floor refinement

HYPOTHESIS: A 13.125% terminal learning-rate floor will exceed 9,240 correct predictions by extending the improvement from 10% to 12.5% while remaining well below the harmful 20% floor.

INTENDED_EDIT: Raise only the cosine schedule’s terminal floor from 12.5% to 13.125% of peak, preserving all other verified behavior.

EVIDENCE: The 12.5% floor achieved the best verified result at 9,240 correct, improving over 9,233 at both 7.5% and 10%, while 20% fell to 9,226; the unverified 13.75% and 15% runs provide no accuracy evidence.

<<<<<<< SEARCH
    multiplier = 0.125 + 0.875 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.13125 + 0.86875 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE