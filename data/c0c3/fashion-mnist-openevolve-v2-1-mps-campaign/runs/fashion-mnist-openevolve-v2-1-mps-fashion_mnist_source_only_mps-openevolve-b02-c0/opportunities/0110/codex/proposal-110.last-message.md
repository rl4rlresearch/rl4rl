MECHANISM: One-sided local temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.3606 to 1.3607 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125956753.

INTENDED_EDIT: Increase only the inference-time ensemble calibration multiplier by 0.0001.

EVIDENCE: Every verified 0.0001 increase from 1.3600 through 1.3606 preserved all 9,253 argmax predictions and reduced cross-entropy; the latest step improved it from 0.2125958950 to 0.2125956753, so the locally improving direction remains upward.

<<<<<<< SEARCH
        return 1.3606 * ensemble_scores
=======
        return 1.3607 * ensemble_scores
>>>>>>> REPLACE