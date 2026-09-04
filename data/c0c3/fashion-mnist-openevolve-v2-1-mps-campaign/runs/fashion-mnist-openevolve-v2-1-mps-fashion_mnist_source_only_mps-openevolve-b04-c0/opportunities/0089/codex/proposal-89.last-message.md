MECHANISM: Bracketed local logit-temperature refinement

HYPOTHESIS: An evaluation-only scale of 1.226016 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.

INTENDED_EDIT: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226016.

EVIDENCE: Scale 1.22603 achieved the best observed cross-entropy, while 1.226032 regressed; a local quadratic fit using the measured results at 1.226, 1.22603, and 1.226032 places the minimum near 1.226016.

<<<<<<< SEARCH
        return 1.22603 * 0.5 * (logits + flipped_logits)
=======
        return 1.226016 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE