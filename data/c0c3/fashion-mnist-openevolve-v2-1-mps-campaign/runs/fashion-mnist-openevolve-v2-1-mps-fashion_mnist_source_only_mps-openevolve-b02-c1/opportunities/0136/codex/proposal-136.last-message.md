MECHANISM: Bracketed logit-ensemble temperature calibration

HYPOTHESIS: Temperature 0.924 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.912 to 0.924.

EVIDENCE: Temperatures 0.900, 0.912, and 1.000 produced identical correct counts with cross-entropies 0.192472, 0.192366, and 0.192871; their fitted minimum lies near 0.933–0.936, so 0.924 is a conservative untested step toward that minimum.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.924)
>>>>>>> REPLACE