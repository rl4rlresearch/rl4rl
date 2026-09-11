MECHANISM: Decision-preserving ensemble temperature sharpening

HYPOTHESIS: An ensemble temperature of 0.75 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.75.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; the two 0.75 attempts timed out without providing performance evidence against the change.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.75
>>>>>>> REPLACE