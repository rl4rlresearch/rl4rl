MECHANISM: Local quadratic logit-temperature calibration

HYPOTHESIS: Restoring the verified 0.04→0 smoothing schedule and scaling evaluation logits by 1.252 will preserve all 9,240 predictions while reducing validation cross-entropy below 0.2167043.

INTENDED_EDIT: Restore the best verified smoothing schedule and move the evaluation-logit multiplier from 1.30 to the estimated local optimum of 1.252.

EVIDENCE: With identical 0.04→0 training, reducing the multiplier from 1.285 to 1.255 preserved 9,240 correct and lowered cross-entropy from 0.2168271 to 0.2167043; together with the approximately 0.216927 result at 1.296352, these measurements place the local calibration optimum near 1.252.

<<<<<<< SEARCH
        return 1.30 * logits
=======
        return 1.252 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE