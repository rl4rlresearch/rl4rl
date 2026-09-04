MECHANISM: Local-quadratic ensemble temperature calibration

HYPOTHESIS: A temperature of 0.717 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2025032.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.723 to 0.717.

EVIDENCE: Cross-entropy improved from 0.2030542 at 0.76 to 0.2026544 at 0.74 and 0.2025032 at 0.723; the declining finite-difference slope places the local quadratic minimum near 0.717, while positive logit scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.723
=======
        return ensemble_log_probabilities / 0.717
>>>>>>> REPLACE