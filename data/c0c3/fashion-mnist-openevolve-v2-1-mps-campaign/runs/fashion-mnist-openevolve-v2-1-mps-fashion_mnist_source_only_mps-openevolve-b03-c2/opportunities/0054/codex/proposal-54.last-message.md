MECHANISM: Cosine-floor peak interpolation

HYPOTHESIS: A 2.4% terminal learning-rate floor will exceed 9,249 correct predictions by approaching the quadratic optimum implied by the 2.0%, 2.2%, and 3.0% results.

INTENDED_EDIT: Raise the cosine schedule’s terminal multiplier from 1.8% to 2.4%, leaving architecture, optimizer, augmentation, and inference unchanged.

EVIDENCE: The 2.2% floor achieved the best result at 9,249 correct, versus 9,243 at 2.0% and 9,235 at 3.0%; quadratic interpolation places the estimated peak near 2.42%.

<<<<<<< SEARCH
    multiplier = 0.018 + 0.982 * cosine
=======
    multiplier = 0.024 + 0.976 * cosine
>>>>>>> REPLACE