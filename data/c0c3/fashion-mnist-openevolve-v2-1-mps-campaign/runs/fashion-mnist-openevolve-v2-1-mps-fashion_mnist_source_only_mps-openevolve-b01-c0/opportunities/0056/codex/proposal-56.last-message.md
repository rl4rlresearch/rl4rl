MECHANISM: Evidence-fitted decision-preserving temperature calibration

HYPOTHESIS: An ensemble temperature of 0.70 will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.2045404.

INTENDED_EDIT: Reduce the final arithmetic-ensemble temperature from 0.80 to 0.70.

EVIDENCE: Temperatures 0.95, 0.90, 0.85, and 0.80 preserved exactly 9,290 correct predictions while lowering cross-entropy to 0.2174928, 0.2119528, 0.2076125, and 0.2045404; the diminishing improvements extrapolate to a calibration minimum near 0.70, while positive logit scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.80
=======
        return ensemble_log_probabilities / 0.70
>>>>>>> REPLACE