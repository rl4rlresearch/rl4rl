MECHANISM: Opposite-direction one-ULP temperature calibration retry

HYPOTHESIS: Lowering the inference temperature by one float32 ULP will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Decrease the positive inference temperature from 0.717143714427948 to 0.7171436548233032.

EVIDENCE: Raising temperature by one ULP preserved 9,290 correct but worsened cross-entropy to 0.20248969802856445; the symmetric lower-temperature probe timed out without metric evidence, so retrying it is the most informative unresolved change.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171436548233032
>>>>>>> REPLACE