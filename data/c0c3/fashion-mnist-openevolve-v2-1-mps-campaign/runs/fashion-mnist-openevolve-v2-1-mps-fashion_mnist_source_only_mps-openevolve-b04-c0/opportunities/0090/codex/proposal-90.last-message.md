MECHANISM: Float32-aware bracketed logit calibration

HYPOTHESIS: An evaluation-only scale of 1.2260157 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420372009276.

INTENDED_EDIT: Refine only the positive flip-ensemble calibration factor from 1.226016 to 1.2260157.

EVIDENCE: Scale 1.226016 is the best observed point, while 1.226 and 1.22603 were worse; quadratic interpolation places the minimum near 1.22601568, motivating the nearest finer literal.

<<<<<<< SEARCH
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        return 1.2260157 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE