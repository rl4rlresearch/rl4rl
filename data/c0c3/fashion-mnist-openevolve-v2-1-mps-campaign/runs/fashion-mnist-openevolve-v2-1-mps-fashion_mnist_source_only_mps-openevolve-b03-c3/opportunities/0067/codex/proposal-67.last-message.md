MECHANISM: Extrapolated logit-temperature recalibration

HYPOTHESIS: With the verified 0.04→0 smoothing schedule, scaling evaluation logits by 1.255 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216827.

INTENDED_EDIT: Restore the best verified smoothing schedule and lower its evaluation-logit multiplier from 1.285 to the estimated calibration optimum of 1.255.

EVIDENCE: Reference Design 1 improved cross-entropy from approximately 0.216927 at scale 1.296352 to 0.216827 at 1.285 without changing correctness; the curvature observed across the original schedule’s calibrated scales indicates that this improvement extrapolates to an optimum near 1.255.

<<<<<<< SEARCH
        return 1.295 * logits
=======
        return 1.255 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE