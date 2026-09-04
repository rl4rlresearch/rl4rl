MECHANISM: Fine-grained ensemble temperature calibration

HYPOTHESIS: Temperature 0.9345 will preserve all 9,311 argmax-correct predictions while reducing validation cross-entropy below 0.1922867332.

INTENDED_EDIT: Raise only the evaluation-time ensemble temperature from 0.9340 to 0.9345.

EVIDENCE: Every verified increase from 0.9300 through 0.9340 reduced cross-entropy without changing predictions; the two 0.9345 attempts timed out and therefore provide no contradictory metric evidence.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.9340)
=======
        return ensemble / (12.0 * 0.9345)
>>>>>>> REPLACE