MECHANISM: Local quadratic temperature refinement

HYPOTHESIS: Raising the inference temperature to 0.7176308860778809 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061935424806.

INTENDED_EDIT: Move the final inference temperature to the nearest float32 value of the quadratic minimum estimated from the three latest informative temperature measurements.

EVIDENCE: Cross-entropy improved from 0.2024606979370117 at 0.717143714427948 to 0.20246065788269044 at 0.717287428855896 and 0.20246061935424806 at 0.717574857711792; fitting these points places the local minimum near 0.71763089, while positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717574857711792
=======
        return ensemble_log_probabilities / 0.7176308860778809
>>>>>>> REPLACE