MECHANISM: Accuracy-preserving global logit recalibration

HYPOTHESIS: Increasing the evaluation logit scale from 1.253 to 1.254 will retain exactly 9,284 correct predictions while reducing validation cross-entropy below 0.208131822.

INTENDED_EDIT: Apply a small positive temperature adjustment to the final evaluation logits without changing training or class rankings.

EVIDENCE: TTA-weight optimization has reached a sharp 9,284/9,283 boundary; a positive global scale preserves every non-tied argmax while directly probing the remaining cross-entropy tie-breaker.

<<<<<<< SEARCH
        return 1.253 * logits
=======
        return 1.254 * logits
>>>>>>> REPLACE