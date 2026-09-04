MECHANISM: One-sided local temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.36 to 1.3601 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2125969967.

INTENDED_EDIT: Increase only the inference-time ensemble calibration multiplier by 0.0001.

EVIDENCE: Decreasing the scale from 1.36 to 1.3599 preserved 9,253 correct predictions but worsened cross-entropy from 0.2125969967 to 0.2125972195, providing local evidence that the improving direction at 1.36 is upward.

<<<<<<< SEARCH
        return 1.36 * ensemble_scores
=======
        return 1.3601 * ensemble_scores
>>>>>>> REPLACE