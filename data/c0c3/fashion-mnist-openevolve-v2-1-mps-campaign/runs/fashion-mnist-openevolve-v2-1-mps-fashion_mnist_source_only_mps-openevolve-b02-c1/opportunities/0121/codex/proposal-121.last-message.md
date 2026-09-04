MECHANISM: Argmax-preserving logit-ensemble recalibration

HYPOTHESIS: Sharpening weighted-logit TTA from temperature 0.912 to 0.900 will preserve all 9,311 correct predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Lower only the evaluation temperature; training, parameters, runtime, and class argmaxes remain unchanged.

EVIDENCE: Temperature 0.912 achieved the same 9,311 correct predictions as temperature 1.0 but lowered cross-entropy from 0.192871 to 0.192366, motivating a small additional sharpening step.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / (12.0 * 0.900)
>>>>>>> REPLACE