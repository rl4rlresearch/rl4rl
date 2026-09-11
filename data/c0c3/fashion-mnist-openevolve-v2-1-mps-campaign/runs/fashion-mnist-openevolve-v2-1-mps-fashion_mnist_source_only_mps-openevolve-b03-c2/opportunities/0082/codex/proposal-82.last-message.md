MECHANISM: Fine-grained quadratic temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.35 to 1.35605 while leaving training and ensemble weighting unchanged.

EVIDENCE: Cross-entropy values at multipliers 1.35, 1.355, and 1.356 imply a local quadratic minimum near 1.356046; all tested positive scalings preserved exactly 9,256 correct predictions.

<<<<<<< SEARCH
        return 1.35 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE