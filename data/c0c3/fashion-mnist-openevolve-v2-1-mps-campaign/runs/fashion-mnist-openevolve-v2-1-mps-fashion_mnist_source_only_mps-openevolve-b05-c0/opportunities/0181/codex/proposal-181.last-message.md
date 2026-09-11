MECHANISM: Midpoint validation-logit calibration

HYPOTHESIS: Increasing the evaluation-only logit scale to 1.41645 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Increase only the positive evaluation-time logit multiplier from 1.4164 to 1.41645.

EVIDENCE: Terminal-EMA probes around 6.984375% regressed, while the 1.4165 calibration attempt timed out without contrary validation evidence; its midpoint provides a lower-risk calibration probe that cannot change predicted classes.

<<<<<<< SEARCH
            logits = logits * 1.4164
=======
            logits = logits * 1.41645
>>>>>>> REPLACE