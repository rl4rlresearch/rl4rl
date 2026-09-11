MECHANISM: One-ULP temperature calibration probe

HYPOTHESIS: Increasing the inference temperature by one float32 representable step will preserve all 9,290 argmax predictions while reducing validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Raise the positive inference temperature from 0.717143714427948 to its next float32 value, 0.7171437740325928.

EVIDENCE: Multiple increasingly fine live/EMA-weight probes tied or worsened the current cross-entropy, indicating that coordinate is quantization-limited; temperature is an independent calibration coordinate whose positive rescaling preserves prediction ordering.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171437740325928
>>>>>>> REPLACE