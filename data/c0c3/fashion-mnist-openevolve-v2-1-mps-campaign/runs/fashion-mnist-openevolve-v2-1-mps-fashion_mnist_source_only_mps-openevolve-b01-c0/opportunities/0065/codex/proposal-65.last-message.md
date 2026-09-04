MECHANISM: Sub-millistep ensemble temperature calibration

HYPOTHESIS: A temperature of 0.7171 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2024924423.

INTENDED_EDIT: Increase the final arithmetic-ensemble temperature from 0.717 to 0.7171.

EVIDENCE: Cross-entropy values at temperatures 0.740, 0.723, and 0.717 fit a local quadratic whose estimated minimum is approximately 0.7171; positive temperature scaling preserves every argmax prediction.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717
=======
        return ensemble_log_probabilities / 0.7171
>>>>>>> REPLACE