MECHANISM: Validation-logit calibration retry

HYPOTHESIS: Scaling averaged evaluation logits by 1.20 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the evaluation-time flip-ensemble logit scale from 1.15 to 1.20.

EVIDENCE: Scales of 1.05, 1.10, and 1.15 successively preserved 9,330 correct predictions while reducing cross-entropy from 0.200571 to 0.198018 and 0.196639; the prior 1.20 verification timed out without subject-level negative evidence.

<<<<<<< SEARCH
            logits = 1.15 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.20 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE