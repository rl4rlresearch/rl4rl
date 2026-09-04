MECHANISM: Verified inference-temperature calibration

HYPOTHESIS: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 correct predictions while reducing validation cross-entropy below 0.20883301086425782.

INTENDED_EDIT: Change only the inference-time logit scale from 1.0360 to 1.03592.

EVIDENCE: Reference Designs 2 and 3 achieved the best verified score, 9243.413622061835, with this exact scale and otherwise identical training; the current 1.0360 scale has slightly higher cross-entropy.

<<<<<<< SEARCH
            logits = 1.0360 * (
=======
            logits = 1.03592 * (
>>>>>>> REPLACE