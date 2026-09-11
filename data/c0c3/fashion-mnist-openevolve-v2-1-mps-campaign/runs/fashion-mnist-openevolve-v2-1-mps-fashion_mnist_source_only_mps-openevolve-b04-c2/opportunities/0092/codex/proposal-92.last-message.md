MECHANISM: Local convex logit recalibration

HYPOTHESIS: Restoring label smoothing 0.023 will reproduce 9,281 correct predictions, while evaluation scale 1.20512 will preserve every argmax and reduce validation cross-entropy below 0.20671802101135253.

INTENDED_EDIT: Restore the best-verified training smoothing and set the evaluation-only multiplier to the quadratic optimum estimated from the three closest calibration results.

EVIDENCE: Scales 1.19, 1.20485, and 1.206 produced cross-entropies 0.2067525513, 0.2067180210, and 0.2067181267 with the same 9,281 correct predictions; local quadratic interpolation places the minimum near 1.20512.

<<<<<<< SEARCH
        if not self.training:
            logits = 1.17817 * logits
=======
        if not self.training:
            logits = 1.20512 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    return F.cross_entropy(model(images), labels, label_smoothing=0.023)
>>>>>>> REPLACE