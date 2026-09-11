MECHANISM: Near-optimal post-ensemble temperature sharpening

HYPOTHESIS: Scaling ensemble scores from 1.30 to 1.35 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21311241.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier from 1.30 to 1.35.

EVIDENCE: Every verified scale increase through 1.30 preserved exactly 9,253 correct predictions and monotonically reduced cross-entropy; finite-difference evidence places the calibration optimum near 1.36, while prior higher-scale runs only timed out and supplied no contrary validation evidence.

<<<<<<< SEARCH
        return 1.30 * ensemble_scores
=======
        return 1.35 * ensemble_scores
>>>>>>> REPLACE