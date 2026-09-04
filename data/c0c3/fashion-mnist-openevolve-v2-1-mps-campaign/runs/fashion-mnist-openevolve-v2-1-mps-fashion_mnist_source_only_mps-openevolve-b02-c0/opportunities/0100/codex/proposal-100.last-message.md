MECHANISM: Evidence-fitted post-ensemble temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.34 to 1.36 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21268247.

INTENDED_EDIT: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Scaling from 1.30 to 1.34 preserved all 9,253 argmax predictions and reduced cross-entropy from 0.21311241 to 0.21268247; prior finite-difference evidence places the calibration optimum near 1.36, while earlier 1.36 attempts timed out without producing contrary validation evidence.

<<<<<<< SEARCH
        return 1.34 * ensemble_scores
=======
        return 1.36 * ensemble_scores
>>>>>>> REPLACE