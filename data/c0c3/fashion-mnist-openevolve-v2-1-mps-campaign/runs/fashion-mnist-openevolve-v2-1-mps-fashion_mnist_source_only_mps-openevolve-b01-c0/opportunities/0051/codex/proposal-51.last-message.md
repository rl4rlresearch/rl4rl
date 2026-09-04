MECHANISM: Near-optimal decision-preserving temperature sharpening

HYPOTHESIS: Reducing ensemble temperature from 0.80 to 0.75 will preserve all 9,290 argmax predictions while lowering validation cross-entropy below 0.2045404.

INTENDED_EDIT: Scale the final arithmetic ensemble log-probabilities by 1/0.75 instead of 1/0.80.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while monotonically reducing cross-entropy; the diminishing improvements suggest 0.75 is a useful next point near the calibration optimum.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.75
>>>>>>> REPLACE