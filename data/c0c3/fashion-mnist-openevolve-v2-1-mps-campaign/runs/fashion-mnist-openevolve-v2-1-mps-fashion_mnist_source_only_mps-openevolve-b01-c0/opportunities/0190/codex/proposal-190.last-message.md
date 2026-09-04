MECHANISM: Uneven-spacing quadratic temperature extrapolation

HYPOTHESIS: Increasing the inference temperature to 0.7180816164016724 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061401367188.

INTENDED_EDIT: Replace only the final ensemble temperature with the float32 value nearest the minimum estimated from the three latest temperature measurements.

EVIDENCE: Cross-entropy improved monotonically from 0.20246065788269044 at 0.717287428855896 to 0.20246061935424806 at 0.717574857711792 and 0.20246061401367188 at 0.7176308860778809; fitting an unevenly spaced quadratic to these points estimates the minimum near 0.7180816, while positive temperature scaling preserves argmax predictions.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.7176308860778809
=======
        return ensemble_log_probabilities / 0.7180816164016724
>>>>>>> REPLACE