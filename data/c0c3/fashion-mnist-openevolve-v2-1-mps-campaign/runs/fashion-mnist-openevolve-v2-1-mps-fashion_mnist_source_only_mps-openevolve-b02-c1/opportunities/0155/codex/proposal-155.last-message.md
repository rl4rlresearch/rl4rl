MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9335 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.1922872028.

INTENDED_EDIT: Raise only the positive evaluation-time ensemble temperature from 0.9325 to 0.9335.

EVIDENCE: Temperatures 0.9300 through 0.9330 successively reduced cross-entropy without changing argmax predictions; the reported 0.9330 cross-entropy of 0.1922872028 motivates the smallest continuation toward the fitted minimum near 0.934.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9325)
=======
        return ensemble / (12.0 * 0.9335)
>>>>>>> REPLACE