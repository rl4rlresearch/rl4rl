MECHANISM: Quadratic confidence-scale calibration

HYPOTHESIS: Setting the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Raise only the final EMA–live ensemble logit scale from 1.04 to the estimated calibration optimum of 1.0495.

EVIDENCE: Scales 1.04, 1.044, and 1.048 successively lowered cross-entropy while preserving 9,251 correct predictions; quadratic interpolation of those measured losses places the minimum near 1.0495, and the prior 1.0495 attempt timed out without contradictory validation evidence.

<<<<<<< SEARCH
            logits = 1.04 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE