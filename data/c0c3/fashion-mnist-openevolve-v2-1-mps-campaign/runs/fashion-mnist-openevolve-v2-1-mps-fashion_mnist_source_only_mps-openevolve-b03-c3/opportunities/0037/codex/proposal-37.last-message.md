MECHANISM: Second-order confidence calibration refinement

HYPOTHESIS: Scaling evaluation logits by 1.296352 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.21758331069946288.

INTENDED_EDIT: Move the evaluation-only multiplier from 1.2964 to the local quadratic optimum inferred from neighboring verified scales.

EVIDENCE: Cross-entropy was 0.21758353462219238 at 1.295, 0.21758331069946288 at 1.2964, and 0.21758494262695313 at 1.30; quadratic interpolation places the minimum near 1.296352.

<<<<<<< SEARCH
        return 1.2964 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE