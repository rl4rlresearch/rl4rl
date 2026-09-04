MECHANISM: Moderate post-ensemble temperature sharpening

HYPOTHESIS: Scaling ensemble scores from 1.281 to 1.30 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21344894.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Every verified scale increase through 1.281 preserved exactly 9,253 correct predictions and monotonically reduced cross-entropy; positive scaling cannot change argmax decisions, while the higher-scale attempts timed out without contrary validation evidence.

<<<<<<< SEARCH
        return 1.281 * ensemble_scores
=======
        return 1.30 * ensemble_scores
>>>>>>> REPLACE