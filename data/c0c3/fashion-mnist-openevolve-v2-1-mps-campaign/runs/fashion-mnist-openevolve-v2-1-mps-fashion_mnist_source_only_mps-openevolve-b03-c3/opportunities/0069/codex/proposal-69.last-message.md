MECHANISM: Three-point quadratic logit-temperature calibration

HYPOTHESIS: Scaling evaluation logits by 1.253 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.2167039406.

INTENDED_EDIT: Increase only the evaluation-logit multiplier from 1.252 to 1.253.

EVIDENCE: With identical 0.04→0 smoothing, scales 1.285, 1.255, and 1.252 produced cross-entropies 0.2168271080, 0.2167042862, and 0.2167039406 while preserving 9,240 correct; their local quadratic fit places the minimum near 1.253.

<<<<<<< SEARCH
        return 1.252 * logits
=======
        return 1.253 * logits
>>>>>>> REPLACE