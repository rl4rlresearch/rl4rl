MECHANISM: Midpoint geometric log-opinion blend

HYPOTHESIS: A 9.5% geometric component will retain 9,267 correct predictions while lowering cross-entropy below 0.215436, producing a strict validation-score improvement.

INTENDED_EDIT: Interpolate 90.5% calibrated arithmetic log-probabilities with 9.5% normalized geometric logits during validation inference.

EVIDENCE: The 9% blend achieved 9,267 correct with monotonically improved cross-entropy, while 10% lost one correct prediction; testing their midpoint is the most informative refinement of this observed decision boundary.

<<<<<<< SEARCH
        return 0.91 * arithmetic_logits + 0.09 * geometric_logits
=======
        return 0.905 * arithmetic_logits + 0.095 * geometric_logits
>>>>>>> REPLACE