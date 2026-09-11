MECHANISM: Conservative post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.20 to 1.25 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2160126.

INTENDED_EDIT: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.

EVIDENCE: Raising the scale from 1.10 to 1.20 preserved all 9,253 predictions and reduced cross-entropy from 0.2224249 to 0.2160126; the 1.30 attempt timed out without producing contrary accuracy or calibration evidence, so 1.25 is a conservative continuation on the best verified design.

<<<<<<< SEARCH
        return 1.20 * ensemble_scores
=======
        return 1.25 * ensemble_scores
>>>>>>> REPLACE