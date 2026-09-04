MECHANISM: Evidence-fitted convex temperature calibration

HYPOTHESIS: Scaling ensemble scores from 1.281 to 1.36 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2134489.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; training, architecture, views, and ensemble weights remain unchanged.

EVIDENCE: Verified scaling from 1.20 through 1.281 preserved exactly 9,253 correct predictions while monotonically reducing cross-entropy. Recent finite differences indicate the convex calibration optimum is near 1.36, while prior higher-scale timeouts supplied no contrary validation evidence.

<<<<<<< SEARCH
        return 1.281 * ensemble_scores
=======
        return 1.36 * ensemble_scores
>>>>>>> REPLACE