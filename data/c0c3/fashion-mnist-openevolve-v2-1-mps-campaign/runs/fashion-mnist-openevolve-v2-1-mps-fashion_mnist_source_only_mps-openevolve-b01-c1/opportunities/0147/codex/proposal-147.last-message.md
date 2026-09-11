MECHANISM: Four-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 3% to 4% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215668, exceeding validation_score 9267.411296.

INTENDED_EDIT: Interpolate 96% calibrated arithmetic log-probabilities with 4% normalized geometric logits during validation inference.

EVIDENCE: Increasing the geometric blend from 1% to 2% raised correct predictions from 9,266 to 9,267, and increasing it to 3% retained 9,267 while reducing cross-entropy from 0.215711 to 0.215668; another one-point step directly tests the established local trend.

<<<<<<< SEARCH
        return 0.97 * arithmetic_logits + 0.03 * geometric_logits
=======
        return 0.96 * arithmetic_logits + 0.04 * geometric_logits
>>>>>>> REPLACE