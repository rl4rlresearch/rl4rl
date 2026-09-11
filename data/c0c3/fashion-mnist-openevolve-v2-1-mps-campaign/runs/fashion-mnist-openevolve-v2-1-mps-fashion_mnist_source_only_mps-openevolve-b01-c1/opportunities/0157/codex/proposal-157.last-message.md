MECHANISM: Binary-refined geometric log-opinion blend

HYPOTHESIS: A 9.875% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215411, yielding a strict validation-score improvement.

INTENDED_EDIT: Interpolate 90.125% calibrated arithmetic log-probabilities with 9.875% normalized geometric logits during validation inference.

EVIDENCE: The 9.75% blend retained 9,267 correct and improved cross-entropy to 0.215411, while 10% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.

<<<<<<< SEARCH
        return 0.9025 * arithmetic_logits + 0.0975 * geometric_logits
=======
        return 0.90125 * arithmetic_logits + 0.09875 * geometric_logits
>>>>>>> REPLACE