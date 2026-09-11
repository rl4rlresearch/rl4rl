MECHANISM: Verified fine-grained ensemble temperature calibration

HYPOTHESIS: Increasing the multiplier to 1.35605 will preserve all 9,256 correct predictions and reduce validation cross-entropy below 0.21445223388671875.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.356 to 1.35605.

EVIDENCE: Reference Design 3 directly verified 1.35605 with 9,256 correct predictions and lower cross-entropy of 0.2144522300720215.

<<<<<<< SEARCH
        return 1.356 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE