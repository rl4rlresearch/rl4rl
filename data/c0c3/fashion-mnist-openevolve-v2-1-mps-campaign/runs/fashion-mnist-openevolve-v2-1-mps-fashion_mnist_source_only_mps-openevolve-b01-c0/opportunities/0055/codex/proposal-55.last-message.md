MECHANISM: Conservative decision-preserving temperature sharpening

HYPOTHESIS: An ensemble temperature of 0.79 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.79.

EVIDENCE: Temperatures from 0.95 through 0.80 preserved exactly 9,290 correct predictions while monotonically lowering cross-entropy; repeated larger reductions timed out, so a minimal step tests whether the established calibration trend continues.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.79
>>>>>>> REPLACE