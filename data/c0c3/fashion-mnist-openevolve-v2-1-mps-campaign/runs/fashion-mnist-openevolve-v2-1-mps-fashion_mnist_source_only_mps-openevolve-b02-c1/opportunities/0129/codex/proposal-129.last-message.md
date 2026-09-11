MECHANISM: Evidence-fitted logit temperature calibration

HYPOTHESIS: Temperature 0.936 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Change only the evaluation-time ensemble temperature from 0.912 to 0.936.

EVIDENCE: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871 respectively; their measured curve places the likely minimum near 0.936.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.936)
>>>>>>> REPLACE