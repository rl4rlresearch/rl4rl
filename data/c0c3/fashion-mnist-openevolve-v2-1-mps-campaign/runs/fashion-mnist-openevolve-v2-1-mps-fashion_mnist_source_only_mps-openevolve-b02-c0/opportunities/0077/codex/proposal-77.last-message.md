MECHANISM: Fine-grained post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.275 to 1.2875 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.

EVIDENCE: Scaling from 1.25 to 1.275 preserved all 9,253 correct predictions and reduced cross-entropy from 0.2142010 to 0.2135743; the smaller intermediate increase probes the continuing calibration trend while avoiding another direct repeat of the twice-timed-out 1.30 setting.

<<<<<<< SEARCH
        return 1.275 * ensemble_scores
=======
        return 1.2875 * ensemble_scores
>>>>>>> REPLACE