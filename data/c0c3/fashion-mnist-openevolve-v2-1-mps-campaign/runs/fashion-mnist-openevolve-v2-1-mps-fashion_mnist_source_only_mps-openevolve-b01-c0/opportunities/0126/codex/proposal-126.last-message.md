MECHANISM: Opposite-direction one-ULP temperature calibration probe

HYPOTHESIS: Lowering the inference temperature by one float32 representable step will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Decrease the positive inference temperature from 0.717143714427948 to 0.7171436548233032.

EVIDENCE: Increasing the temperature by one ULP preserved 9,290 correct predictions but worsened cross-entropy to 0.20248969802856445, making the symmetric lower-temperature probe the most informative remaining calibration change.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171436548233032
>>>>>>> REPLACE