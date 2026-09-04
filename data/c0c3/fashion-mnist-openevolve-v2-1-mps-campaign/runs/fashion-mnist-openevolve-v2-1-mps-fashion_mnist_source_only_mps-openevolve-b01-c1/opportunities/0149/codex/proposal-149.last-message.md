MECHANISM: Six-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 5% to 6% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215586, exceeding validation_score 9267.411324.

INTENDED_EDIT: Interpolate 94% calibrated arithmetic log-probabilities with 6% normalized geometric logits during validation inference.

EVIDENCE: Every increase from 1% through 5% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 5% result makes another one-point step the most direct test of this trend.

<<<<<<< SEARCH
        return 0.95 * arithmetic_logits + 0.05 * geometric_logits
=======
        return 0.94 * arithmetic_logits + 0.06 * geometric_logits
>>>>>>> REPLACE