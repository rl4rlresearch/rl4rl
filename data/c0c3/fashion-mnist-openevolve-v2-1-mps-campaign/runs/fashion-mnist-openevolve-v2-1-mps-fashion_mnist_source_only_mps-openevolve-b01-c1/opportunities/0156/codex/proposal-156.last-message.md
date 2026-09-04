MECHANISM: Three-quarter boundary geometric log-opinion blend

HYPOTHESIS: A 9.75% geometric component will retain 9,267 correct predictions while lowering cross-entropy below 0.215419, yielding a strict validation-score improvement.

INTENDED_EDIT: Interpolate 90.25% calibrated arithmetic log-probabilities with 9.75% normalized geometric logits during validation inference.

EVIDENCE: The 9.5% blend retained 9,267 correct while improving cross-entropy to 0.215419, whereas 10% lost one correct; testing 9.75% is the most informative refinement of this observed decision boundary.

<<<<<<< SEARCH
        return 0.905 * arithmetic_logits + 0.095 * geometric_logits
=======
        return 0.9025 * arithmetic_logits + 0.0975 * geometric_logits
>>>>>>> REPLACE