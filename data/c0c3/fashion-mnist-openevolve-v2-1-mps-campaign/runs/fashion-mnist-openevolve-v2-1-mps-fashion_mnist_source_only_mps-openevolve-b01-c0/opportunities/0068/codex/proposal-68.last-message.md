MECHANISM: Local quadratic micro-calibration

HYPOTHESIS: A temperature of 0.71714 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20249243392944336.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.71715 to 0.71714.

EVIDENCE: Cross-entropy at temperatures 0.7170, 0.7171, and 0.71715 was 0.20249244232177735, 0.20249243469238282, and 0.20249243392944336; a local quadratic fit places the minimum near 0.717144, while positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71715
=======
        return ensemble_log_probabilities / 0.71714
>>>>>>> REPLACE