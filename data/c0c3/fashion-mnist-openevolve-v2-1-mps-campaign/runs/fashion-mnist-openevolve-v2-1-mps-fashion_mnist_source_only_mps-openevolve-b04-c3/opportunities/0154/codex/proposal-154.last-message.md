MECHANISM: Verified confidence-scale calibration

HYPOTHESIS: Raising the inference scale from 1.048 to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Change only the final EMA–live ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.

EVIDENCE: Reference Design 1 verified this exact implementation at 9,251 correct with cross-entropy 0.20655155601501465, strictly improving the current design at the same correct count.

<<<<<<< SEARCH
            logits = 1.048 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE