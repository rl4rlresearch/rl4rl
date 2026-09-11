MECHANISM: Intermediate decision-preserving temperature sharpening

HYPOTHESIS: An ensemble temperature of 0.775 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.775.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; 0.775 tests a smaller step toward the repeatedly timed-out 0.75 setting.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.775
>>>>>>> REPLACE