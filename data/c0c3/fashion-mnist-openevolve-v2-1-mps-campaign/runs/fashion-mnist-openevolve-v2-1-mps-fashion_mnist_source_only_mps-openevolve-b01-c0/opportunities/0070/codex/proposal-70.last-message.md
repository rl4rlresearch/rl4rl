MECHANISM: Local-quadratic optimum temperature calibration

HYPOTHESIS: A temperature of 0.71714375 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20249243392944336.

INTENDED_EDIT: Replace the final ensemble temperature 0.71715 with the fitted quadratic optimum 0.71714375.

EVIDENCE: Cross-entropies at 0.7170, 0.7171, and 0.71715 place the fitted minimum at 0.71714375; the nearby 0.717144 timeout provided no contrary metric evidence, and positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71715
=======
        return ensemble_log_probabilities / 0.71714375
>>>>>>> REPLACE