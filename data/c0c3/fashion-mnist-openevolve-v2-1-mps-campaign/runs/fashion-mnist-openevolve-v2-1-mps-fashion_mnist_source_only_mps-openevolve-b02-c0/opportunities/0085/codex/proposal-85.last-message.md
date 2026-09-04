MECHANISM: Fine-grained post-ensemble temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.28 to 1.285 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2134692.

INTENDED_EDIT: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: Increasing the scale from 1.275 to 1.28 preserved all 9,253 predictions and reduced cross-entropy from 0.2135743 to 0.2134692. Positive scaling preserves argmax predictions, and higher-scale timeouts supplied no contrary accuracy or calibration evidence.

<<<<<<< SEARCH
        return 1.28 * ensemble_scores
=======
        return 1.285 * ensemble_scores
>>>>>>> REPLACE