MECHANISM: Fine-grained post-ensemble temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.28 to 1.2825 will preserve exactly 9,253 predictions while reducing validation cross-entropy below 0.2134692.

INTENDED_EDIT: Change only the inference-time calibration multiplier; retain the verified training, architecture, views, and ensemble weights.

EVIDENCE: The verified increase from 1.275 to 1.28 preserved all 9,253 correct predictions and lowered cross-entropy from 0.2135743 to 0.2134692; positive scaling preserves argmax, and the timed-out 1.285 run supplied no contrary score evidence.

<<<<<<< SEARCH
        return 1.28 * ensemble_scores
=======
        return 1.2825 * ensemble_scores
>>>>>>> REPLACE