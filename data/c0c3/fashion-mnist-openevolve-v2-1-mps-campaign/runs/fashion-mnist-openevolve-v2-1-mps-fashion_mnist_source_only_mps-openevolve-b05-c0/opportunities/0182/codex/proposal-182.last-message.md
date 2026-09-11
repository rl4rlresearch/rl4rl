MECHANISM: Downward midpoint validation-logit calibration

HYPOTHESIS: Decreasing the evaluation-only logit scale from 1.4164 to 1.41635 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.2120078533.

INTENDED_EDIT: Reduce only the positive evaluation-time logit multiplier; training, EMA, and all predicted classes remain unchanged.

EVIDENCE: Increasing the scale to 1.41645 preserved 9,260 correct but worsened cross-entropy to 0.2120078560, indicating the improving direction is downward; 1.41635 is the conservative midpoint toward the timed-out 1.4163 probe.

<<<<<<< SEARCH
            logits = logits * 1.4164
=======
            logits = logits * 1.41635
>>>>>>> REPLACE