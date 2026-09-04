MECHANISM: Evidence-fitted post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores from 1.30 to 1.36 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21311241.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Verified scaling through 1.30 preserved exactly 9,253 correct predictions and monotonically reduced cross-entropy; recent finite differences place the convex calibration optimum near 1.36, and the prior 1.36 timeout supplied no contrary validation evidence.

<<<<<<< SEARCH
        return 1.30 * ensemble_scores
=======
        return 1.36 * ensemble_scores
>>>>>>> REPLACE