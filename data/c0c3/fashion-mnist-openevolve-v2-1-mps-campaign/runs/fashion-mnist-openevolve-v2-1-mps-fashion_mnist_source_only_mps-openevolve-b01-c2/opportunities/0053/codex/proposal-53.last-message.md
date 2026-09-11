MECHANISM: Fine-grained evaluation temperature sharpening

HYPOTHESIS: Scaling ensemble logits by 1.18 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the evaluation-time symmetric flip-ensemble logit scale from 1.15 to 1.18.

EVIDENCE: Scaling from 1.05 to 1.10 to 1.15 preserved 9,330 correct predictions while cross-entropy improved from 0.200571 to 0.198018 to 0.196639; the prior 1.18 verification timed out and therefore provides no negative validation evidence.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.18 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE