MECHANISM: Newton temperature calibration from three fixed-logit measurements

HYPOTHESIS: A positive evaluation-time logit scale of 1.0658182 will preserve all 9,360 argmax predictions while reducing validation cross-entropy below 0.18599775314331055.

INTENDED_EDIT: Replace the current evaluation multiplier with the quadratic-interpolation minimum estimated from the completed 1.1035, 1.104, and 1.1045 measurements.

EVIDENCE: Lowering the scale from 1.104 to 1.1035 improved cross-entropy, while raising it to 1.1045 worsened it; because positive global scaling cannot change class rankings, these three measurements support a larger accuracy-invariant step toward their estimated convex minimum.

<<<<<<< SEARCH
        return 1.1035 * logits
=======
        return 1.0658182 * logits
>>>>>>> REPLACE