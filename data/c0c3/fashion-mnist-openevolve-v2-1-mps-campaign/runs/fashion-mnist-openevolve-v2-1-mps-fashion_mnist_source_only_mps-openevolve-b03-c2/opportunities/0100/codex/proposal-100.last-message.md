MECHANISM: Verified ensemble temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions and reduce validation cross-entropy from 0.2147829205 to approximately 0.2144522301.

INTENDED_EDIT: Replace only the evaluation-time logit multiplier, leaving training and ensemble weighting unchanged.

EVIDENCE: Reference Design 2 directly verified this exact implementation at 9,256 correct with cross-entropy 0.2144522300720215, a strict improvement over the current design.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE