MECHANISM: Conservative logit-ensemble temperature sharpening

HYPOTHESIS: Temperature 0.900 will preserve all 9,311 argmax predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Lower only the evaluation-time ensemble temperature from 0.912 to 0.900.

EVIDENCE: Temperature 0.912 preserved unit temperature’s 9,311 correct predictions while improving cross-entropy from 0.192871 to 0.192366; prior 0.900 attempts timed out and produced no contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.900)
>>>>>>> REPLACE