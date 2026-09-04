MECHANISM: Seven-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 6% to 7% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215547, exceeding validation_score 9267.411338.

INTENDED_EDIT: Interpolate 93% calibrated arithmetic log-probabilities with 7% normalized geometric logits during validation inference.

EVIDENCE: Every increase from 1% through 6% geometric blending preserved or improved validation correct while monotonically lowering cross-entropy; the successful 6% result makes another one-point step the clearest test of the established trend.

<<<<<<< SEARCH
        return 0.94 * arithmetic_logits + 0.06 * geometric_logits
=======
        return 0.93 * arithmetic_logits + 0.07 * geometric_logits
>>>>>>> REPLACE