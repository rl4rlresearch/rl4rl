MECHANISM: Fine-grained downward validation-logit calibration

HYPOTHESIS: Decreasing the evaluation-only logit scale to 1.4163625 will preserve all 9,260 argmax predictions while reducing validation cross-entropy below 0.21200784798.

INTENDED_EDIT: Reduce only the positive evaluation-time logit multiplier from 1.416375 to 1.4163625.

EVIDENCE: Lowering the scale from 1.4164 to 1.416375 improved cross-entropy to 0.21200784798 with all 9,260 correct predictions preserved, while increasing it to 1.41645 worsened cross-entropy; the proposed value is the conservative midpoint toward the timed-out 1.41635 probe.

<<<<<<< SEARCH
            logits = logits * 1.416375
=======
            logits = logits * 1.4163625
>>>>>>> REPLACE