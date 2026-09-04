MECHANISM: Verified symmetric-ensemble temperature sharpening

HYPOTHESIS: Scaling evaluation logits by 1.15 will preserve all 9,330 argmax predictions while reducing validation cross-entropy from 0.200571 to approximately the verified 0.196639.

INTENDED_EDIT: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.05 to 1.15.

EVIDENCE: Reference Design 2 used the identical training procedure with scale 1.15 and achieved 9,330 correct with 0.196639 cross-entropy, strictly outperforming the current 9,330 correct with 0.200571 cross-entropy.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.15 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE