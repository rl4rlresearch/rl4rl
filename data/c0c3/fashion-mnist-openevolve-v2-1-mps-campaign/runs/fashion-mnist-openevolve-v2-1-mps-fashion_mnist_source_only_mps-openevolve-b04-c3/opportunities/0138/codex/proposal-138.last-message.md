MECHANISM: Quadratic confidence-scale calibration

HYPOTHESIS: Setting the positive inference scale to 1.05 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Raise only the final EMA–live ensemble logit scale from 1.04 to 1.05, leaving training and class rankings unchanged.

EVIDENCE: Successive scales of 1.04, 1.044, and 1.048 preserved 9,251 correct while reducing cross-entropy to 0.20656522, 0.20655614, and 0.20655191; the shrinking gains indicate an optimum near 1.05.

<<<<<<< SEARCH
            logits = 1.04 * logits
=======
            logits = 1.05 * logits
>>>>>>> REPLACE