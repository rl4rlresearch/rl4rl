MECHANISM: Verified confidence-scale calibration

HYPOTHESIS: Increasing the positive inference scale from 1.048 to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy below 0.20655190887451172.

INTENDED_EDIT: Raise only the final EMA–live ensemble logit scale, leaving training, rankings, parameter count, and runtime characteristics unchanged.

EVIDENCE: Reference Design 1 verified scale 1.0495 at 9,251 correct with cross-entropy 0.20655155601501465, strictly improving the current scale 1.048 result of 0.20655190887451172 without changing correctness.

<<<<<<< SEARCH
            logits = 1.048 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE