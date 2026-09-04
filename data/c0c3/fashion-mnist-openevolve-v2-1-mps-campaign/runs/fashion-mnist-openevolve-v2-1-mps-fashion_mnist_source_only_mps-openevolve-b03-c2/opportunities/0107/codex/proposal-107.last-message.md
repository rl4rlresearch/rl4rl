MECHANISM: Verified ensemble temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2145212341 to approximately 0.2144522301.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.33 to 1.35605.

EVIDENCE: Reference Design 2 directly verified this otherwise-identical implementation at 9,256 correct and 0.2144522300720215 cross-entropy, strictly improving the current validation score; failed verification attempts reported no contradictory measurement.

<<<<<<< SEARCH
        return 1.33 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE