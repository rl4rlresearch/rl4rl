MECHANISM: Bracketed quadratic temperature interpolation

HYPOTHESIS: Setting the inference temperature to 0.7177127828598023 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.

INTENDED_EDIT: Replace only the final ensemble temperature with the nearest float32 value to the quadratic minimum bracketed by the latest temperature measurements.

EVIDENCE: Cross-entropy was higher at both 0.717574857711792 and 0.7180816164016724 than at 0.7176308860778809; fitting those three measurements places the minimum near 0.7177128, while positive temperature scaling preserves predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.7176308860778809
=======
        return ensemble_log_probabilities / 0.7177127828598023
>>>>>>> REPLACE