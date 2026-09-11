MECHANISM: Tight-bracket quadratic temperature interpolation

HYPOTHESIS: Setting the inference temperature to 0.7176485061645508 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Replace only the final inference temperature with the nearest float32 value to the quadratic minimum estimated from the three tightest measurements.

EVIDENCE: The current 0.717663049697876 achieved the best cross-entropy, while both 0.7176308860778809 and 0.7176686525344849 were worse; quadratic interpolation of this bracket estimates a minimum near 0.7176485, and positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities / 0.7176485061645508
>>>>>>> REPLACE