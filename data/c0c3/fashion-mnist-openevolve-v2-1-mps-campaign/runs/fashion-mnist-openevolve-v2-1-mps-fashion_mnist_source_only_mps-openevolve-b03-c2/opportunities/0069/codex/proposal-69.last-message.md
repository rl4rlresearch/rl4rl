MECHANISM: Incremental argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.06 will retain 9,256 correct predictions while reducing validation cross-entropy below 0.2306832.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.03 to 1.06, leaving training and inference weighting unchanged.

EVIDENCE: Scaling from 1.00 to 1.03 preserved 9,256 correct predictions and reduced cross-entropy from 0.2347709 to 0.2306832; another equal-sized step tests whether the ensemble remains underconfident.

<<<<<<< SEARCH
        return 1.03 * aggregate_logits
=======
        return 1.06 * aggregate_logits
>>>>>>> REPLACE