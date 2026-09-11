MECHANISM: Two-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 1% to 2% will retain 9,266 correct predictions while lowering cross-entropy below 0.215755, exceeding validation_score 9266.411267.

INTENDED_EDIT: Interpolate 98% calibrated arithmetic log-probabilities with 2% normalized geometric logits during validation inference.

EVIDENCE: Adding a 1% geometric component preserved the arithmetic ensemble’s 9,266 correct predictions and improved cross-entropy from 0.215801 to 0.215755; the geometric endpoint’s substantially lower 0.211944 cross-entropy motivates one more conservative step in the same direction.

<<<<<<< SEARCH
        return 0.99 * arithmetic_logits + 0.01 * geometric_logits
=======
        return 0.98 * arithmetic_logits + 0.02 * geometric_logits
>>>>>>> REPLACE