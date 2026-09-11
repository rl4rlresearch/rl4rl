MECHANISM: Nonzero-floor cosine upper-bound refinement

HYPOTHESIS: A 3% terminal learning-rate floor will retain the late-training stability of the best 2% floor while avoiding slight over-settling, exceeding 9,243 correct predictions.

INTENDED_EDIT: Change the cosine schedule’s terminal multiplier from 10% to 3%, leaving architecture, optimizer, augmentation, and inference unchanged.

EVIDENCE: The 2% floor achieved 9,243 correct, outperforming both the 1% floor’s 9,239 and zero floor’s 9,209; testing 3% probes the untested upper neighbor of the established optimum with no runtime cost.

<<<<<<< SEARCH
    multiplier = 0.10 + 0.90 * cosine
=======
    multiplier = 0.03 + 0.97 * cosine
>>>>>>> REPLACE