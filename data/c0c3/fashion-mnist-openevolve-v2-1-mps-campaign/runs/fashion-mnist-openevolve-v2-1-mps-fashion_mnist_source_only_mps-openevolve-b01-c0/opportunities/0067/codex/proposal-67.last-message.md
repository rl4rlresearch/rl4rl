MECHANISM: Midpoint micro-bracketing temperature calibration

HYPOTHESIS: A temperature of 0.71715 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.2024924347.

INTENDED_EDIT: Increase the final arithmetic-ensemble temperature from 0.7171 to 0.71715.

EVIDENCE: Raising the temperature from 0.7170 to 0.7171 reduced cross-entropy by 7.63e-9 without changing any prediction, while the fitted minimum is approximately 0.71717; 0.71715 is a conservative midpoint below the timed-out 0.7172 trial.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.7171
=======
        return ensemble_log_probabilities / 0.71715
>>>>>>> REPLACE