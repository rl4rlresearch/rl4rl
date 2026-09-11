MECHANISM: Evidence-fitted ensemble temperature calibration

HYPOTHESIS: Temperature 0.933 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Change only the evaluation-time ensemble temperature from 0.912 to 0.933.

EVIDENCE: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; quadratic interpolation places the minimum near 0.933, while prior 0.933 verifications timed out without contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.933)
>>>>>>> REPLACE