MECHANISM: Moderate post-ensemble logit sharpening

HYPOTHESIS: Scaling ensemble log-probabilities by 1.15 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2287977.

INTENDED_EDIT: Increase the inference-only positive ensemble scale from 1.05 to 1.15.

EVIDENCE: Raising the scale from 1.00 to 1.05 preserved 9,252 correct and reduced cross-entropy from 0.2356299 to 0.2287977; larger-scale verifications timed out without supplying contrary performance evidence, so a moderate step provides a more informative calibration test than another fine increment.

<<<<<<< SEARCH
        return 1.05 * ensemble_log_probabilities
=======
        return 1.15 * ensemble_log_probabilities
>>>>>>> REPLACE