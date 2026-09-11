MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.74 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2030542.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.76 to 0.74.

EVIDENCE: Temperatures from 0.95 through 0.76 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; the diminishing improvements place the calibration minimum near 0.73–0.74, and positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.76
=======
        return ensemble_log_probabilities / 0.74
>>>>>>> REPLACE