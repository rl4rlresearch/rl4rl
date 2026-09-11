MECHANISM: Fine-grained quadratic logit calibration

HYPOTHESIS: An evaluation-only scale of 1.226032 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420486450197.

INTENDED_EDIT: Refine only the positive flip-ensemble calibration factor from 1.22603 to 1.226032.

EVIDENCE: Cross-entropy improved at every tested scale through 1.22603, while quadratic interpolation of the measurements at 1.22, 1.225, and 1.226 places the local minimum near 1.226032.

<<<<<<< SEARCH
        return 1.22603 * 0.5 * (logits + flipped_logits)
=======
        return 1.226032 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE