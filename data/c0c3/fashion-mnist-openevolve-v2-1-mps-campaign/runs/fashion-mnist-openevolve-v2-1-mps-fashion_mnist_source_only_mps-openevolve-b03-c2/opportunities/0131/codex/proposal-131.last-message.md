MECHANISM: Verified evaluation-temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy to 0.2144522300720215.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.35641 to 1.35605.

EVIDENCE: Reference Design 2 verified the otherwise-identical implementation at 9,256 correct with lower cross-entropy and a strictly higher validation_score than the current design; unsuccessful attempts reported no contradictory measurement.

<<<<<<< SEARCH
        return 1.35641 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE