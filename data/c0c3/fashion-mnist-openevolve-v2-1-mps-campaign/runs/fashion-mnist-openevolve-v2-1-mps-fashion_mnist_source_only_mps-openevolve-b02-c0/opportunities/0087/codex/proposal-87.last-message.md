MECHANISM: Conservative post-ensemble temperature sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.28 to 1.281 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.2134692.

INTENDED_EDIT: Change only the inference-time calibration multiplier; retain the verified architecture, training procedure, views, and ensemble weights.

EVIDENCE: The verified increase from 1.275 to 1.28 preserved 9,253 correct predictions and lowered cross-entropy from 0.2135743 to 0.2134692. Positive scaling preserves argmax, and the timed-out higher-scale attempts provided no contrary validation evidence.

<<<<<<< SEARCH
        return 1.28 * ensemble_scores
=======
        return 1.281 * ensemble_scores
>>>>>>> REPLACE