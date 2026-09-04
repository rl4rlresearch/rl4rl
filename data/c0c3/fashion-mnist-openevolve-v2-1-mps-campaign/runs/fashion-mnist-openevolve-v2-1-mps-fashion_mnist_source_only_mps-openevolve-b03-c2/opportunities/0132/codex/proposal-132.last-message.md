MECHANISM: Verified evaluation-temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2147829205 to the verified 0.2144522301.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.30 to 1.35605.

EVIDENCE: Reference Design 2 is otherwise identical and achieved 9,256 correct with the best verified cross-entropy; unsuccessful attempts supplied no contradictory measurement.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE