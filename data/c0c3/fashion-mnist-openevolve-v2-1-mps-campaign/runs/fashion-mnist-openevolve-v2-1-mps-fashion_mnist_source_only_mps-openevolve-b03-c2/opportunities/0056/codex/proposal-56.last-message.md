MECHANISM: Updated cosine-floor vertex refinement

HYPOTHESIS: A 2.25% terminal learning-rate floor will exceed 9,249 correct predictions by targeting the quadratic vertex implied by the verified 2.15%, 2.2%, and 2.4% results.

INTENDED_EDIT: Lower the current cosine schedule floor from 3% to 2.25%, leaving architecture, optimizer, augmentation, and inference unchanged.

EVIDENCE: The 2.2% floor achieved the best result at 9,249 correct, versus 9,242 at 2.15% and 9,232 at 2.4%; quadratic interpolation of these closest observations places the estimated peak near 2.25%.

<<<<<<< SEARCH
    multiplier = 0.03 + 0.97 * cosine
=======
    multiplier = 0.0225 + 0.9775 * cosine
>>>>>>> REPLACE