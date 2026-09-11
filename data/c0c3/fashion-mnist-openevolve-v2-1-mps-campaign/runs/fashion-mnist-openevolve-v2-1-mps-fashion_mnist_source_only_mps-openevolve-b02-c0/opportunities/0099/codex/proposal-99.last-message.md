MECHANISM: Evidence-fitted post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores from 1.30 to 1.34 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21311241.

INTENDED_EDIT: Increase only the positive inference-time calibration multiplier from 1.30 to 1.34.

EVIDENCE: Every verified scale increase through 1.30 preserved 9,253 correct predictions and reduced cross-entropy; finite-difference evidence estimates the optimum near 1.36, while higher-scale attempts timed out without contrary validation evidence.

<<<<<<< SEARCH
        return 1.30 * ensemble_scores
=======
        return 1.34 * ensemble_scores
>>>>>>> REPLACE