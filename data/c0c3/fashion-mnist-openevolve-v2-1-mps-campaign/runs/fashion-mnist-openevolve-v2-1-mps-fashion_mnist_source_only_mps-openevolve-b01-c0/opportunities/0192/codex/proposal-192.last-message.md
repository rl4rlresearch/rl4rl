MECHANISM: Bracketed local quadratic temperature refinement

HYPOTHESIS: Setting the inference temperature to 0.717663049697876 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.

INTENDED_EDIT: Replace only the final inference temperature with the nearest float32 value to the minimum estimated from the tightest three-point bracket.

EVIDENCE: Cross-entropy is higher at both 0.717574857711792 and 0.7177127828598023 than at 0.7176308860778809; quadratic interpolation of those measurements estimates the minimum near 0.717663, while positive temperature scaling leaves predictions unchanged.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.7176308860778809
=======
        return ensemble_log_probabilities / 0.717663049697876
>>>>>>> REPLACE