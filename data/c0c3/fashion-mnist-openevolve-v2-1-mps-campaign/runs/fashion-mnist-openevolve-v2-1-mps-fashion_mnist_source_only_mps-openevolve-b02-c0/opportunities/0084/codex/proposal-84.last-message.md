MECHANISM: Evidence-fitted post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores from 1.28 to 1.35 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2134692.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; training, augmentation, views, and ensemble weights remain unchanged.

EVIDENCE: Every verified scale increase through 1.28 preserved 9,253 correct and reduced cross-entropy; the recent 1.275-to-1.28 improvement shows sharpening remains beneficial, while positive scaling preserves argmax and prior higher-scale timeouts provide no contrary score evidence.

<<<<<<< SEARCH
        return 1.28 * ensemble_scores
=======
        return 1.35 * ensemble_scores
>>>>>>> REPLACE