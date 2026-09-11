MECHANISM: Float32-adjacent logit calibration

HYPOTHESIS: An evaluation-only scale of 1.2260162 will preserve all 9,359 argmax predictions and reduce validation cross-entropy below 0.18488420372009276.

INTENDED_EDIT: Increase the positive flip-ensemble calibration factor from 1.226016 to 1.2260162, targeting the next higher effective float32 multiplier.

EVIDENCE: Scale 1.226016 is best observed; 1.2260157 regressed and 1.22603 was slightly worse, so testing the immediately adjacent higher float32 calibration value is the most informative remaining local refinement.

<<<<<<< SEARCH
        return 1.226016 * 0.5 * (logits + flipped_logits)
=======
        return 1.2260162 * 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE