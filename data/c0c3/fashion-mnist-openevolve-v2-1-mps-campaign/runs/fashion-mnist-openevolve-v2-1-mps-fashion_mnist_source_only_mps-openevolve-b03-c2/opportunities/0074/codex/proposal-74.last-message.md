MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.21 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2182493.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.12 to 1.21, leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers from 1.03 through 1.18 preserved exactly 9,256 correct predictions while monotonically lowering cross-entropy; 1.18 achieved the best observed 0.2182493, and its positive marginal improvement supports one more 0.03 step.

<<<<<<< SEARCH
        return 1.12 * aggregate_logits
=======
        return 1.21 * aggregate_logits
>>>>>>> REPLACE