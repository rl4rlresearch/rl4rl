MECHANISM: Quadratic-fit ensemble temperature calibration

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.355 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2144559.

INTENDED_EDIT: Replace the current 1.18 post-ensemble multiplier with the estimated calibration optimum of 1.355.

EVIDENCE: Cross-entropy improved from 0.2147829 at 1.30 to 0.2145212 at 1.33 and 0.2144559 at 1.35; the diminishing gains imply a local quadratic minimum near 1.355, while positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.18 * aggregate_logits
=======
        return 1.355 * aggregate_logits
>>>>>>> REPLACE