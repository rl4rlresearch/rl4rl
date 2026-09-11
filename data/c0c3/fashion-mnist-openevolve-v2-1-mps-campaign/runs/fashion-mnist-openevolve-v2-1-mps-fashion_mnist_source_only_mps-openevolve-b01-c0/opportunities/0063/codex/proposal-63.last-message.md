MECHANISM: Local-quadratic ensemble temperature calibration

HYPOTHESIS: A temperature of 0.723 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2026544.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.74 to the evidence-fitted estimate 0.723.

EVIDENCE: Cross-entropy fell from 0.2045404 at 0.80 to 0.2030542 at 0.76 and 0.2026544 at 0.74; fitting the diminishing local improvements estimates the calibration minimum near 0.723, while positive logit scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.74
=======
        return ensemble_log_probabilities / 0.723
>>>>>>> REPLACE