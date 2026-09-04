MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.24 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2169634.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.09 to 1.24 while leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers from 1.03 through 1.21 preserved exactly 9,256 correct predictions and monotonically reduced cross-entropy; the latest 1.21 result achieved 0.2169634, with a positive marginal improvement supporting another 0.03 step.

<<<<<<< SEARCH
        return 1.09 * aggregate_logits
=======
        return 1.24 * aggregate_logits
>>>>>>> REPLACE