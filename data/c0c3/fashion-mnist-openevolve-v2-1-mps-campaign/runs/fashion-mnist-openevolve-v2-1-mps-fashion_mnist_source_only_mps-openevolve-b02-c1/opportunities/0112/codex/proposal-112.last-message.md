MECHANISM: Logit-ensemble temperature recalibration

HYPOTHESIS: Using unit temperature for weighted logit TTA will preserve exactly 9,311 correct predictions while reducing validation cross-entropy below 0.192366.

INTENDED_EDIT: Increase the evaluation temperature from 0.912 to 1.0 without changing training or class decisions.

EVIDENCE: Weighted logit averaging produced the current best 9,311 correct while retaining the 0.912 temperature from probability aggregation; logit averaging is typically sharper, and positive temperature scaling preserves every argmax.

<<<<<<< SEARCH
        return ensemble / (12.0 * 0.912)
=======
        return ensemble / 12.0
>>>>>>> REPLACE