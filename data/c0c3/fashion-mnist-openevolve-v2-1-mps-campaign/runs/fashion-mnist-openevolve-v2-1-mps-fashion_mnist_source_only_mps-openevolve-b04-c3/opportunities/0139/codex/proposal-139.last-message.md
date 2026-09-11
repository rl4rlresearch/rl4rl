MECHANISM: Quadratic confidence-scale interpolation

HYPOTHESIS: Setting the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while lowering validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Change only the final EMA–live ensemble logit scale from 1.03592 to the estimated calibration optimum of 1.0495.

EVIDENCE: Scales 1.04, 1.044, and 1.048 successively reduced cross-entropy while preserving 9,251 correct; quadratic interpolation of those measured losses places the minimum near 1.0495, while the unverified 1.05 attempt supplies no contradictory result.

<<<<<<< SEARCH
            logits = 1.03592 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE