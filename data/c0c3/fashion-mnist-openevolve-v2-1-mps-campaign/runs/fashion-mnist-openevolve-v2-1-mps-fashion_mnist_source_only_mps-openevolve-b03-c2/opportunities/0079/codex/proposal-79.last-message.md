MECHANISM: Near-optimal argmax-preserving temperature sharpening

HYPOTHESIS: Scaling the verified 3.25:1 ensemble logits by 1.35 will retain exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2145212.

INTENDED_EDIT: Increase the post-ensemble logit multiplier from 1.21 to 1.35 while leaving training and ensemble weighting unchanged.

EVIDENCE: Scaling from 1.30 to 1.33 preserved all 9,256 correct predictions and lowered cross-entropy from 0.2147829 to 0.2145212; the sharply diminishing gains place the likely calibration optimum near 1.35.

<<<<<<< SEARCH
        return 1.21 * aggregate_logits
=======
        return 1.35 * aggregate_logits
>>>>>>> REPLACE