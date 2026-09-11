MECHANISM: Evidence-fitted ensemble temperature calibration

HYPOTHESIS: Temperature 0.936 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Change only the positive evaluation-time logit scaling factor from 0.912 to 0.936.

EVIDENCE: Temperatures 0.900, 0.912, and 1.000 produced the same 9,311 correct predictions with cross-entropies 0.192472, 0.192366, and 0.192871; interpolation places the minimum near 0.936, whose prior verification timed out without contradictory metrics.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.936)
>>>>>>> REPLACE