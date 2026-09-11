MECHANISM: Incremental post-ensemble calibration sharpening

HYPOTHESIS: Increasing the positive ensemble scale from 1.20 to 1.30 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2160126.

INTENDED_EDIT: Change only the inference-time scale applied to the verified arithmetic–geometric ensemble.

EVIDENCE: Raising the scale from 1.10 to 1.20 preserved all 9,253 predictions and reduced cross-entropy from 0.2224249 to 0.2160126; another positive scale increase cannot change finite-logit argmax predictions and tests whether the ensemble remains underconfident.

<<<<<<< SEARCH
        return 1.20 * ensemble_scores
=======
        return 1.30 * ensemble_scores
>>>>>>> REPLACE