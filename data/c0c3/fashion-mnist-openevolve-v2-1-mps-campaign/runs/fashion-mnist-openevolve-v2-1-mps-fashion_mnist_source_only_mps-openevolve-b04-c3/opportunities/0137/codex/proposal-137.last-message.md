MECHANISM: One-step confidence calibration extrapolation

HYPOTHESIS: Increasing the positive inference scale from 1.044 to 1.048 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655614318847657.

INTENDED_EDIT: Raise only the final EMA–live ensemble logit scale, leaving training, class rankings, and parameter count unchanged.

EVIDENCE: Scale increases from 1.03592 to 1.04 and then 1.044 both preserved 9,251 correct predictions while successively reducing cross-entropy, indicating the ensemble remains slightly under-confident.

<<<<<<< SEARCH
            logits = 1.044 * logits
=======
            logits = 1.048 * logits
>>>>>>> REPLACE