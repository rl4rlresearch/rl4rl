MECHANISM: Fine-grained post-ensemble temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.275 to 1.28 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Change only the inference-time calibration multiplier, leaving training, views, and ensemble decisions unchanged.

EVIDENCE: Every verified increase from 1.10 through 1.275 preserved 9,253 correct predictions and monotonically lowered cross-entropy; positive scaling preserves argmax, while the later timeouts supplied no contrary calibration evidence.

<<<<<<< SEARCH
        return 1.275 * ensemble_scores
=======
        return 1.28 * ensemble_scores
>>>>>>> REPLACE