MECHANISM: Nonzero-floor cosine boundary refinement

HYPOTHESIS: A 1% terminal learning-rate floor will retain the stability lost at zero while further reducing late-update noise, exceeding 9,243 correct predictions.

INTENDED_EDIT: Change the cosine schedule’s terminal multiplier from 10% to 1%, leaving the verified architecture, optimizer, augmentation, and inference unchanged.

EVIDENCE: The 2% floor achieved the best result at 9,243 correct, while zero fell to 9,209; testing the midpoint isolates whether a smaller strictly positive floor improves the established optimum.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * cosine
=======
    multiplier = 0.01 + 0.99 * cosine
>>>>>>> REPLACE