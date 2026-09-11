MECHANISM: Verified ensemble temperature calibration

HYPOTHESIS: Raising the post-ensemble multiplier to 1.35605 will preserve all 9,256 correct predictions and reduce validation cross-entropy from 0.2147829205 to approximately 0.2144522301.

INTENDED_EDIT: Replace only the evaluation-time ensemble logit multiplier, leaving training and ensemble weighting unchanged.

EVIDENCE: Reference Design 2 directly verified 1.35605 with 9,256 correct predictions and the best available cross-entropy, 0.2144522300720215; prior positive scaling preserved argmax predictions.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE