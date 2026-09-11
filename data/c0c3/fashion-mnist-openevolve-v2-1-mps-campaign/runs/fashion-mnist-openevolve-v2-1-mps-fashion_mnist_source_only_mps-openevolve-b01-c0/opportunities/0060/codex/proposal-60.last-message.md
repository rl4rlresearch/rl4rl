MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.72 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.74 to 0.72.

EVIDENCE: Temperature reductions through 0.74 preserved exactly 9,290 correct predictions and monotonically reduced cross-entropy; the smaller improvement from 0.76 to 0.74 suggests the calibration minimum is nearby, motivating a conservative 0.02 step.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.74
=======
        return ensemble_log_probabilities / 0.72
>>>>>>> REPLACE