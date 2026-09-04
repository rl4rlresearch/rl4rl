MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.18 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2198722.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.15 to 1.18, leaving training and ensemble weighting unchanged.

EVIDENCE: Multipliers from 1.03 through 1.15 consistently preserved 9,256 correct predictions while progressively lowering cross-entropy; the latest 1.15 result reached 0.2198722, supporting another equal sharpening step.

<<<<<<< SEARCH
        return 1.15 * aggregate_logits
=======
        return 1.18 * aggregate_logits
>>>>>>> REPLACE