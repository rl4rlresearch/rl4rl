MECHANISM: Continued argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.33 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2147829.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.24 to 1.33 while leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers from 1.03 through 1.30 preserved exactly 9,256 correct predictions and monotonically lowered cross-entropy; the verified 1.30 design achieved the best result at 0.2147829, and its remaining positive marginal improvement motivates the next 0.03 step.

<<<<<<< SEARCH
        return 1.24 * aggregate_logits
=======
        return 1.33 * aggregate_logits
>>>>>>> REPLACE