MECHANISM: Fine-grained evaluation temperature sharpening

HYPOTHESIS: Scaling ensemble logits by 1.18 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.18.

EVIDENCE: Cross-entropy improved from 0.200571 at scale 1.05 to 0.198018 at 1.10 and 0.196639 at 1.15 without changing correctness; the diminishing gains imply a calibration optimum near 1.18.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.18 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE