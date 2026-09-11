MECHANISM: One-sided local temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.3602 to 1.3603 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125965469.

INTENDED_EDIT: Increase only the inference-time ensemble calibration multiplier by 0.0001.

EVIDENCE: The consecutive increases from 1.36 to 1.3601 and then 1.3602 preserved all 9,253 argmax predictions while reducing cross-entropy from 0.2125969967 to 0.2125967724 and then 0.2125965469; positive scaling preserves argmax decisions, and the locally improving direction remains upward.

<<<<<<< SEARCH
        return 1.3602 * ensemble_scores
=======
        return 1.3603 * ensemble_scores
>>>>>>> REPLACE