MECHANISM: Verified fine-grained ensemble temperature calibration

HYPOTHESIS: Increasing the multiplier to 1.35605 will preserve all 9,256 correct predictions and reduce validation cross-entropy below 0.2144522339.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.356 to 1.35605.

EVIDENCE: Reference Design 3 directly verified 1.35605 with 9,256 correct predictions and cross-entropy 0.2144522300720215, a strict improvement over the current 1.356 result.

<<<<<<< SEARCH
        return 1.356 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE