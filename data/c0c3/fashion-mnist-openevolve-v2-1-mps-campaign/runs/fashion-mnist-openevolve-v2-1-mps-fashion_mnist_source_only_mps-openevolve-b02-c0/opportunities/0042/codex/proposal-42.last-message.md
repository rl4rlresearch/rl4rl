MECHANISM: Conservative post-ensemble logit sharpening

HYPOTHESIS: Scaling ensemble log-probabilities by 1.075 will preserve exactly 9,252 correct predictions while reducing validation cross-entropy below 0.2287977.

INTENDED_EDIT: Increase the inference-only ensemble scale from 1.05 to 1.075, midway toward the inconclusive 1.10 attempt.

EVIDENCE: Increasing the scale from 1.00 to 1.05 preserved 9,252 correct and lowered cross-entropy from 0.2356299 to 0.2287977; the 1.10 verification timed out without performance evidence, motivating a conservative intermediate scale.

<<<<<<< SEARCH
        return 1.05 * ensemble_log_probabilities
=======
        return 1.075 * ensemble_log_probabilities
>>>>>>> REPLACE