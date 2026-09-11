MECHANISM: Incremental post-ensemble temperature sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.30 to 1.31 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21311241.

INTENDED_EDIT: Change only the inference-time calibration multiplier, retaining the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Scaling from 1.281 to 1.30 preserved all 9,253 argmax predictions and reduced cross-entropy from 0.21344894 to 0.21311241; positive scaling preserves argmax, and the observed calibration trend remains improving toward the estimated optimum near 1.36.

<<<<<<< SEARCH
        return 1.30 * ensemble_scores
=======
        return 1.31 * ensemble_scores
>>>>>>> REPLACE