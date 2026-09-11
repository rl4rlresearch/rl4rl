MECHANISM: Binary-refined geometric log-opinion blend

HYPOTHESIS: A 9.84375% geometric component will retain 9,267 correct predictions while lowering validation cross-entropy below 0.215409, yielding a strict validation-score improvement.

INTENDED_EDIT: Interpolate 90.15625% calibrated arithmetic log-probabilities with 9.84375% normalized geometric logits during validation inference.

EVIDENCE: The 9.8125% blend retained 9,267 correct, whereas 9.875% lost one correct prediction; their midpoint is the most informative refinement of the observed decision boundary.

<<<<<<< SEARCH
        return 0.901875 * arithmetic_logits + 0.098125 * geometric_logits
=======
        return 0.9015625 * arithmetic_logits + 0.0984375 * geometric_logits
>>>>>>> REPLACE