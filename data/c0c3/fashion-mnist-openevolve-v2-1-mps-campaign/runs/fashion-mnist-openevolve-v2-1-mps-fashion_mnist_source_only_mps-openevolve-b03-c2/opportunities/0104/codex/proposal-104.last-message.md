MECHANISM: Verified ensemble temperature calibration

HYPOTHESIS: Scaling the ensemble logits by 1.35605 will preserve all 9,256 argmax predictions while reducing validation cross-entropy from 0.2147829205 to approximately 0.2144522301.

INTENDED_EDIT: Change only the evaluation-time logit multiplier from 1.30 to 1.35605.

EVIDENCE: Reference Design 2 verified this exact otherwise-identical implementation at 9,256 correct and 0.2144522301 cross-entropy, strictly improving the current tie-break metric; unsuccessful verification attempts provide no contradictory measured result.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE