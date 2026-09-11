MECHANISM: One-sided micro-bracketing temperature calibration

HYPOTHESIS: A temperature of 0.7172 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924347.

INTENDED_EDIT: Increase the final arithmetic-ensemble temperature from 0.7171 to 0.7172.

EVIDENCE: Increasing temperature from 0.7170 to 0.7171 preserved all predictions and reduced cross-entropy by 7.63e-9; combined with the broader local curvature, this places the estimated minimum near 0.71717 and motivates testing the adjacent upper point.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.7171
=======
        return ensemble_log_probabilities / 0.7172
>>>>>>> REPLACE