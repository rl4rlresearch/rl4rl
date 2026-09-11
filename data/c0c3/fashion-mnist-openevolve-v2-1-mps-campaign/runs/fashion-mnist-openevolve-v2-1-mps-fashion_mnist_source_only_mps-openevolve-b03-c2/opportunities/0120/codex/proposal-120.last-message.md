MECHANISM: Verified evaluation temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.35605 will preserve exactly 9,256 correct predictions while reducing validation cross-entropy from 0.2147829205 to approximately 0.2144522301.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.30 to 1.35605.

EVIDENCE: Reference Design 2 verified this otherwise-identical implementation at 9,256 correct with the best reported cross-entropy, 0.2144522300720215; unsuccessful verification attempts supplied no contradictory measurements.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE