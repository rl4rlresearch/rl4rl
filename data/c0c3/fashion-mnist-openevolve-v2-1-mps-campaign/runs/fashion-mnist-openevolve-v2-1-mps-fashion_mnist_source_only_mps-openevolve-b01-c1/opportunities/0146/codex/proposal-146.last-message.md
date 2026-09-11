MECHANISM: Three-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 2% to 3% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215711, thereby exceeding validation_score 9267.411282.

INTENDED_EDIT: Interpolate 97% calibrated arithmetic log-probabilities with 3% normalized geometric logits during validation inference.

EVIDENCE: Moving from 1% to 2% geometric blending improved validation correct from 9,266 to 9,267 and reduced cross-entropy from 0.215755 to 0.215711; the geometric endpoint also has substantially lower cross-entropy, motivating one further conservative step.

<<<<<<< SEARCH
        return 0.98 * arithmetic_logits + 0.02 * geometric_logits
=======
        return 0.97 * arithmetic_logits + 0.03 * geometric_logits
>>>>>>> REPLACE