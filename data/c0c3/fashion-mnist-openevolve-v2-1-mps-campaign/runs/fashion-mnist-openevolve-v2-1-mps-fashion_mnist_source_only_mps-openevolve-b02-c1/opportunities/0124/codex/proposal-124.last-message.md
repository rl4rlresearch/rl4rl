MECHANISM: Argmax-preserving logit-ensemble recalibration

HYPOTHESIS: Temperature 0.900 will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Lower only the evaluation temperature from 0.912 to 0.900.

EVIDENCE: Temperature 0.912 matched unit temperature’s 9,311 correct predictions while lowering cross-entropy from 0.192871 to 0.192366; the prior 0.900 attempt timed out and therefore did not test the predicted further sharpening benefit.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.900)
>>>>>>> REPLACE