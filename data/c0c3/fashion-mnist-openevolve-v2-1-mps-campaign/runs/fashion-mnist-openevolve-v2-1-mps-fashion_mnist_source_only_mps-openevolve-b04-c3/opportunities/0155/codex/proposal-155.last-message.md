MECHANISM: Verified confidence-scale calibration

HYPOTHESIS: Raising the inference scale to 1.0495 will preserve exactly 9,251 correct predictions while reducing validation cross-entropy from 0.20657955322265625 to approximately 0.20655155601501465.

INTENDED_EDIT: Change only the final ensemble logit scale, leaving training, class rankings, parameters, and runtime characteristics unchanged.

EVIDENCE: Reference Design 1 verified this exact implementation at 9,251 correct and 0.20655155601501465 cross-entropy, strictly outperforming the current design at the same correct count; failed verification attempts provide no contradictory measurement.

<<<<<<< SEARCH
            logits = 1.03592 * logits
=======
            logits = 1.0495 * logits
>>>>>>> REPLACE