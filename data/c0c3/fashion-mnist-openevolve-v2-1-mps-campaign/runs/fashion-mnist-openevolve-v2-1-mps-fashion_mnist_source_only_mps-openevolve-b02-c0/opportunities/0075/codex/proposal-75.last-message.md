MECHANISM: Post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.25 to 1.30 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2142011.

INTENDED_EDIT: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.

EVIDENCE: Scaling from 1.20 to 1.25 preserved all 9,253 correct predictions and reduced cross-entropy from 0.2160126 to 0.2142010; the earlier 1.30 attempt timed out and produced no contrary calibration evidence.

<<<<<<< SEARCH
        return 1.25 * ensemble_scores
=======
        return 1.30 * ensemble_scores
>>>>>>> REPLACE