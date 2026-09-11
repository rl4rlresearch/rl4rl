MECHANISM: Quadratic-fit validation-logit calibration

HYPOTHESIS: Scaling averaged evaluation logits by 1.184 will preserve all 9,330 argmax predictions while reducing validation cross-entropy below 0.196639.

INTENDED_EDIT: Increase only the evaluation-time flip-ensemble logit scale from 1.05 to the estimated calibration optimum of 1.184.

EVIDENCE: Scales 1.05, 1.10, and 1.15 preserved 9,330 correct while reducing cross-entropy to 0.200571, 0.198018, and 0.196639; their quadratic fit places the minimum near 1.184, and the prior 1.185 run timed out without negative validation evidence.

<<<<<<< SEARCH
            logits = 1.05 * (0.5 * (logits + flipped_logits))
=======
            logits = 1.184 * (0.5 * (logits + flipped_logits))
>>>>>>> REPLACE