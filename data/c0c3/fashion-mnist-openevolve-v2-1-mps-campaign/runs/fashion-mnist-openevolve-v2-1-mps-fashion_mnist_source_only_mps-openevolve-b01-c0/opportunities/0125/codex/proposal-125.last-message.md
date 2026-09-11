MECHANISM: One-ULP temperature calibration retry

HYPOTHESIS: Raising the inference temperature to the next float32 value will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20248969497680663.

INTENDED_EDIT: Increase the inference temperature from 0.717143714427948 to 0.7171437740325928.

EVIDENCE: Fine live/EMA mixture probes only tied or worsened cross-entropy, while the sole independent one-ULP temperature probe timed out without metric evidence; retrying it is the most informative unresolved change.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717143714427948
=======
        return ensemble_log_probabilities / 0.7171437740325928
>>>>>>> REPLACE