MECHANISM: Conservative post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.25 to 1.275 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2142011.

INTENDED_EDIT: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.

EVIDENCE: Scaling from 1.20 to 1.25 preserved all 9,253 correct predictions and reduced cross-entropy from 0.2160126 to 0.2142010; both 1.30 attempts timed out without calibration evidence, motivating a smaller intermediate step.

<<<<<<< SEARCH
        return 1.25 * ensemble_scores
=======
        return 1.275 * ensemble_scores
>>>>>>> REPLACE