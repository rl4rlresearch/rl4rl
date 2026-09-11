MECHANISM: Downward logit-temperature recalibration

HYPOTHESIS: Scaling evaluation logits by 1.285 will preserve all 9,240 argmax predictions while reducing validation cross-entropy below 0.216927 by compensating for the greater confidence induced by the new 0.04→0 smoothing schedule.

INTENDED_EDIT: Reduce only the evaluation-logit multiplier from 1.296352 to 1.285.

EVIDENCE: The 1.296352 optimum was measured with the reference 0.05→0.01 smoothing schedule, whereas the current 0.04→0 schedule lowers smoothing by 0.01 throughout training and improved cross-entropy to 0.216927 without changing correctness; a modest downward recalibration tests that confidence shift while mathematically preserving every predicted class.

<<<<<<< SEARCH
        return 1.296352 * logits
=======
        return 1.285 * logits
>>>>>>> REPLACE