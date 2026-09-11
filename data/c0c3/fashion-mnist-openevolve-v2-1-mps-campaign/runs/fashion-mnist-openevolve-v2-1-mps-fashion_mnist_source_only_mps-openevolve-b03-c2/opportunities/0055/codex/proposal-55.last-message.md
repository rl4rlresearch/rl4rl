MECHANISM: Cosine-floor vertex refinement

HYPOTHESIS: A 2.15% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic peak implied by the verified 2.0%, 2.2%, and 2.4% results.

INTENDED_EDIT: Change the cosine schedule’s terminal multiplier from 1.0% to 2.15%, leaving architecture, optimizer, augmentation, and inference unchanged.

EVIDENCE: The 2.2% floor achieved 9,249 correct, versus 9,243 at 2.0% and 9,232 at 2.4%; quadratic interpolation of these three observations places the estimated peak near 2.15%.

<<<<<<< SEARCH
    multiplier = 0.01 + 0.99 * cosine
=======
    multiplier = 0.0215 + 0.9785 * cosine
>>>>>>> REPLACE