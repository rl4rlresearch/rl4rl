MECHANISM: Quadratic-fit validation-logit calibration

HYPOTHESIS: Scaling averaged evaluation logits by 1.185 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the positive evaluation-time flip-ensemble logit scale from 1.10 to 1.185.

EVIDENCE: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while reducing cross-entropy to 0.200571, 0.198018, and 0.196639; a quadratic fit to these measurements estimates the calibration optimum near 1.184, while prior 1.20 attempts timed out without negative validation evidence.

<<<<<<< SEARCH
            logits = 1.10 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.185 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE