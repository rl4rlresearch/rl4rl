MECHANISM: Verified flip-ensemble logit calibration

HYPOTHESIS: Scaling evaluation logits by 1.15 will preserve the current 9,330 correct predictions while reducing validation cross-entropy from 0.198018 to approximately 0.196639.

INTENDED_EDIT: Increase only the evaluation-time flip-ensemble logit scale from 1.10 to the best verified value, 1.15.

EVIDENCE: Reference Design 3 used the identical training procedure with scale 1.15, retained 9,330 correct predictions, and achieved the best verified validation cross-entropy of 0.196639.

<<<<<<< SEARCH
            logits = 1.10 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE