MECHANISM: Verified inference-temperature calibration

HYPOTHESIS: Restoring the verified 1.03592 inference scale will preserve 9,243 correct predictions and reduce cross-entropy below the current 0.20883301391601564.

INTENDED_EDIT: Change only the flip-ensemble logit scale from 1.03577 to 1.03592.

EVIDENCE: Reference Design 3 achieved the highest verified score, 9243.413622061835, with this exact scale and otherwise identical code; the current 1.03577 scale scored slightly lower.

<<<<<<< SEARCH
            logits = 1.03577 * (
=======
            logits = 1.03592 * (
>>>>>>> REPLACE