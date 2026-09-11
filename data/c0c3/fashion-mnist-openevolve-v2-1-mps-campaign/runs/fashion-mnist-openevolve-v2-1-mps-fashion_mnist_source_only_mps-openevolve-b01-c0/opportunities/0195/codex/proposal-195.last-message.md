MECHANISM: Local quadratic temperature interpolation

HYPOTHESIS: Setting the inference temperature to 0.7176592946052551 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Replace only the final inference temperature with the nearest float32 value to the quadratic minimum estimated from the three closest measurements around the current best.

EVIDENCE: The current 0.717663049697876 has lower cross-entropy than both 0.7176485061645508 and 0.7176686525344849; interpolating those three results estimates an interior minimum near 0.7176593, while positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities / 0.7176592946052551
>>>>>>> REPLACE