MECHANISM: Verified fine-grained ensemble temperature calibration

HYPOTHESIS: Increasing the multiplier from 1.356 to 1.35605 will preserve exactly 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

INTENDED_EDIT: Replace only the post-ensemble logit multiplier with the best verified value.

EVIDENCE: Reference Design 3 verified multiplier 1.35605 at 9,256 correct and cross-entropy 0.2144522300720215, a strict improvement over the current 1.356 result.

<<<<<<< SEARCH
        return 1.356 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE