MECHANISM: Deeper terminal learning-rate annealing

HYPOTHESIS: Lowering the cosine schedule’s final learning rate from 9e-5 to 3e-5 will exceed 9,262 correct predictions by reducing late-update noise while preserving the successful early trajectory.

INTENDED_EDIT: Change only the cosine learning-rate floor from 3% to 1% of the 3e-3 peak.

EVIDENCE: The best architecture and 0.020→0.009 smoothing schedule remain strongest, while capacity changes regressed and late EMA fell to 9,244 correct; a more convergent endpoint directly tests late optimization stability without extra parameters, steps, or runtime-heavy averaging.

<<<<<<< SEARCH
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
=======
        multiplier = 0.01 + 0.99 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
>>>>>>> REPLACE