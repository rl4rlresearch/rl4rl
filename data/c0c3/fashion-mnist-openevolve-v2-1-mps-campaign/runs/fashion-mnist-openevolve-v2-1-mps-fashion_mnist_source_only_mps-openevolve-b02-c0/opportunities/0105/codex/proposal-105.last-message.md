MECHANISM: One-sided local temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.3601 to 1.3602 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2125967724.

INTENDED_EDIT: Increase only the inference-time ensemble calibration multiplier by 0.0001.

EVIDENCE: Moving from 1.36 to 1.3601 reduced cross-entropy from 0.2125969967 to 0.2125967724, while moving downward to 1.3599 worsened it to 0.2125972195. This brackets 1.36 with consistent evidence that the locally improving direction remains upward; positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.3601 * ensemble_scores
=======
        return 1.3602 * ensemble_scores
>>>>>>> REPLACE