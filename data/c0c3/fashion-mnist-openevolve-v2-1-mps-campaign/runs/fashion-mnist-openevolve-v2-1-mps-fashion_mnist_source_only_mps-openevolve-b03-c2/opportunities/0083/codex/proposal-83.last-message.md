MECHANISM: Verified ensemble temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy from 0.2145212341 to approximately 0.2144522301.

INTENDED_EDIT: Replace the current 1.33 post-ensemble multiplier with the best verified multiplier, leaving training and ensemble weighting unchanged.

EVIDENCE: Reference Design 2 directly verified 1.35605 with 9,256 correct predictions and the best available cross-entropy, 0.2144522300720215.

<<<<<<< SEARCH
        return 1.33 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE