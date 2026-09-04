MECHANISM: Verified inference-temperature calibration

HYPOTHESIS: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 correct predictions while reducing validation cross-entropy below 0.20883301086425782.

INTENDED_EDIT: Change only the inference-time logit scale from 1.0360 to the best verified value, leaving training and class rankings unchanged.

EVIDENCE: Reference Design 3 achieved the highest verified score, 9243.413622061835, with scale 1.03592 and cross-entropy 0.20883300514221192 using the otherwise identical implementation.

<<<<<<< SEARCH
            logits = 1.0360 * (
=======
            logits = 1.03592 * (
>>>>>>> REPLACE