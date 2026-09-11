MECHANISM: Float32-grid quadratic temperature refinement

HYPOTHESIS: Setting the inference temperature to 0.7176634073257446 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Increase only the final inference temperature by six float32 ULPs to the nearest representable value of the local quadratic minimum.

EVIDENCE: The current temperature outperformed probes 63 ULPs lower and 94 ULPs higher; quadratic interpolation of their cross-entropy increases estimates the minimum approximately 6.4 ULPs above the current value.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities / 0.7176634073257446
>>>>>>> REPLACE