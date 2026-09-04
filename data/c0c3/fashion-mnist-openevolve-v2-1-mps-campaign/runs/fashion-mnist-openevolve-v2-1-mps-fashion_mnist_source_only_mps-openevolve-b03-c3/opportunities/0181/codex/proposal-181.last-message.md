MECHANISM: Quadratic-refined global logit calibration

HYPOTHESIS: An evaluation scale of 1.1645 will preserve all 9,284 argmax predictions while lowering validation cross-entropy below 0.207286140.

INTENDED_EDIT: Increase only the positive evaluation-logit multiplier from 1.153 to the curvature-refined optimum of approximately 1.1645.

EVIDENCE: Scale reductions near 1.248 showed measurable convex curvature, while the large jump to 1.153 improved cross-entropy to 0.207286140; fitting those observations places the estimated minimum near 1.1645, and positive global scaling preserves class rankings.

<<<<<<< SEARCH
        return 1.153 * logits
=======
        return 1.1645 * logits
>>>>>>> REPLACE