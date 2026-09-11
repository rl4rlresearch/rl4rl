MECHANISM: Local quadratic evaluation-temperature calibration

HYPOTHESIS: Scaling ensemble logits by 1.356157 will preserve exactly 9,256 argmax predictions and reduce validation cross-entropy below 0.2144522300720215.

INTENDED_EDIT: Change only the evaluation-time ensemble logit multiplier from 1.35605 to 1.356157.

EVIDENCE: Multipliers 1.356, 1.35605, and 1.35641 yielded cross-entropies 0.21445223388671875, 0.2144522300720215, and 0.21445224533081056 with identical correct counts; quadratic interpolation places the minimum near 1.356157.

<<<<<<< SEARCH
        return 1.35605 * aggregate_logits
=======
        return 1.356157 * aggregate_logits
>>>>>>> REPLACE