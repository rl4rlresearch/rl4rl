MECHANISM: Verified inference-temperature calibration

HYPOTHESIS: Scaling the flip-ensemble logits by 1.03592 will preserve all 9,243 class predictions while reducing validation cross-entropy from 0.20883301086425782 to approximately 0.20883300514221192.

INTENDED_EDIT: Change only the positive inference-time logit scale from 1.0360 to 1.03592.

EVIDENCE: Reference Designs 2 and 3 achieved the highest verified score, 9243.413622061835, using this exact scale with otherwise identical code; prior timeouts provide no contradictory performance evidence.

<<<<<<< SEARCH
            logits = 1.0360 * (
=======
            logits = 1.03592 * (
>>>>>>> REPLACE