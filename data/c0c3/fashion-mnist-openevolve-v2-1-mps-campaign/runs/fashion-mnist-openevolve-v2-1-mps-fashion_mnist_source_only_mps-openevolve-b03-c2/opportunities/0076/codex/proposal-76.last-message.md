MECHANISM: Continued argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.27 will retain exactly 9,256 correct predictions while reducing validation cross-entropy below 0.2159778.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.06 to 1.27 while leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers through 1.24 preserved exactly 9,256 correct predictions and monotonically reduced cross-entropy; 1.24 achieved the best observed 0.2159778, and its positive marginal gain motivates the next 0.03 step.

<<<<<<< SEARCH
        return 1.06 * aggregate_logits
=======
        return 1.27 * aggregate_logits
>>>>>>> REPLACE