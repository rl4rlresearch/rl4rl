MECHANISM: Five-percent geometric log-opinion blend

HYPOTHESIS: Increasing the geometric component from 4% to 5% will retain at least 9,267 correct predictions while lowering cross-entropy below 0.215626, exceeding validation_score 9267.411311.

INTENDED_EDIT: Interpolate 95% calibrated arithmetic log-probabilities with 5% normalized geometric logits during validation inference.

EVIDENCE: Each increase from 1% through 4% geometric blending preserved or improved validation correct while monotonically reducing cross-entropy; the 4% blend achieved the current best result, so one further one-point step is the clearest continuation of the established local trend.

<<<<<<< SEARCH
        return 0.96 * arithmetic_logits + 0.04 * geometric_logits
=======
        return 0.95 * arithmetic_logits + 0.05 * geometric_logits
>>>>>>> REPLACE