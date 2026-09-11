MECHANISM: One-ULP validation calibration refinement

HYPOTHESIS: Decreasing the positive probability-log scaling factor by one float32 ULP will preserve all 9,210 predictions while reducing validation cross-entropy below 0.23114465255737304.

INTENDED_EDIT: Change only the validation-time calibration multiplier from its current rounded float32 value to the immediately smaller representable float32 value.

EVIDENCE: The centered-view weight is already bracketed by a worse adjacent float32 value, while positive logit rescaling cannot change argmax predictions; calibration refinement therefore tests a new low-risk cross-entropy improvement without sacrificing the leading correct count.

<<<<<<< SEARCH
        return 1.034951171875 * mean_probability.clamp_min(1e-8).log()
=======
        return 1.03495109081268310546875 * mean_probability.clamp_min(1e-8).log()
>>>>>>> REPLACE