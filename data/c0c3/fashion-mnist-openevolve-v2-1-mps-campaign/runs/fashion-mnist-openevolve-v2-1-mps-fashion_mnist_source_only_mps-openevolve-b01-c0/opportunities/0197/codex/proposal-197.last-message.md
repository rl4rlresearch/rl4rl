MECHANISM: One-ULP temperature grid search

HYPOTHESIS: Increasing the inference temperature by one float32 ULP to 0.7176631093025208 will preserve all 9,290 argmax predictions and reduce validation cross-entropy below 0.20246061096191406.

INTENDED_EDIT: Increase only the final inference temperature to the nearest representable value above the current best.

EVIDENCE: The current temperature is better than probes 63 ULPs below and 6 ULPs above; the prior quadratic estimate favored the upward direction, so the adjacent upper float is the most informative untested refinement.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717663049697876
=======
        return ensemble_log_probabilities / 0.7176631093025208
>>>>>>> REPLACE