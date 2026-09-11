MECHANISM: Continued argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.30 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2152603.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.27 to 1.30 while leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers from 1.03 through 1.27 preserved exactly 9,256 correct predictions and monotonically lowered cross-entropy; the latest 1.27 result reached 0.2152603, and its positive marginal improvement supports another 0.03 step.

<<<<<<< SEARCH
        return 1.27 * aggregate_logits
=======
        return 1.30 * aggregate_logits
>>>>>>> REPLACE