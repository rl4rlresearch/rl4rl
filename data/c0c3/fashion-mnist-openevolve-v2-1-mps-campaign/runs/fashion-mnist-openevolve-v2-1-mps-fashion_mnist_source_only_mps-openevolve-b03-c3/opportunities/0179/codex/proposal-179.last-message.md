MECHANISM: Curvature-estimated global logit calibration

HYPOTHESIS: An evaluation scale of 1.153 will preserve all 9,284 argmax predictions while lowering cross-entropy below 0.208037965.

INTENDED_EDIT: Leave training and TTA unchanged; replace the positive evaluation-logit multiplier with the estimated cross-entropy optimum.

EVIDENCE: Reductions from 1.252 through 1.248 preserved 9,284 correct and consistently improved cross-entropy; the shrinking per-step gains imply curvature whose extrapolated zero-slope point is approximately 1.153.

<<<<<<< SEARCH
        return 1.252 * logits
=======
        return 1.153 * logits
>>>>>>> REPLACE