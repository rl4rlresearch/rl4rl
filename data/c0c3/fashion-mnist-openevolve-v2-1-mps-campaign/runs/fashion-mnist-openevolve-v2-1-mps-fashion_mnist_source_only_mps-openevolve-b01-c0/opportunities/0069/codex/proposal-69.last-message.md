MECHANISM: Local-quadratic optimum temperature calibration

HYPOTHESIS: A temperature of 0.717144 will preserve all 9,290 argmax predictions and produce lower validation cross-entropy than 0.20249243392944336.

INTENDED_EDIT: Replace the final arithmetic-ensemble temperature 0.71715 with the fitted optimum 0.717144.

EVIDENCE: Cross-entropies at 0.7170, 0.7171, and 0.71715 fit a quadratic minimum near 0.71714375; the 0.71714 timeout supplied no contrary performance evidence, and positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.71715
=======
        return ensemble_log_probabilities / 0.717144
>>>>>>> REPLACE