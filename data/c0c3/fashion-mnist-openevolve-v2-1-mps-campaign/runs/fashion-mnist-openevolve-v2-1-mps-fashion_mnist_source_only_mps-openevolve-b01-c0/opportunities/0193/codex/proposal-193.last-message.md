MECHANISM: Tight-bracket quadratic temperature refinement

HYPOTHESIS: Setting the inference temperature to 0.7176686525344849 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Replace only the final inference temperature with the nearest float32 value to the quadratic minimum estimated from the tightest available bracket.

EVIDENCE: Cross-entropy is higher at both 0.7176308860778809 and 0.7177127828598023 than at the current best 0.717663049697876; quadratic interpolation of these three measurements places the minimum near 0.71766865, while positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities / 0.7176686525344849
>>>>>>> REPLACE