MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.73 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.74 to 0.73.

EVIDENCE: Reductions through 0.74 preserved exactly 9,290 correct predictions and monotonically lowered cross-entropy; improvements are diminishing toward an estimated minimum near 0.72, while the 0.72 timeout supplied no contrary accuracy or calibration evidence.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.74
=======
        return ensemble_log_probabilities / 0.73
>>>>>>> REPLACE