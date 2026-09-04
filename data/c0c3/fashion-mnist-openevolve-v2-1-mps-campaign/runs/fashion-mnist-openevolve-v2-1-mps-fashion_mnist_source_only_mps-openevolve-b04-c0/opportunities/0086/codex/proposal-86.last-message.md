MECHANISM: Quadratic-refined evaluation logit calibration

HYPOTHESIS: An evaluation-only scale of 1.226 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488433990478514.

INTENDED_EDIT: Increase only the positive flip-ensemble calibration factor from 1.225 to 1.226.

EVIDENCE: Cross-entropy was 0.1849714141845703 at 1.20, 0.18488877143859864 at 1.22, and 0.18488433990478514 at 1.225; quadratic interpolation of these results places the local minimum near 1.226.

<<<<<<< SEARCH
        return 1.225 * 0.5 * (logits + flipped_logits)
=======
        return 1.226 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE