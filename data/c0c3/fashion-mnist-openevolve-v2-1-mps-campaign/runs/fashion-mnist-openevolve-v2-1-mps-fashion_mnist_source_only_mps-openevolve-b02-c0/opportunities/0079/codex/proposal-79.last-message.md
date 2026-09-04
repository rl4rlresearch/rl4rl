MECHANISM: Evidence-guided post-ensemble calibration sharpening

HYPOTHESIS: Scaling the verified ensemble scores from 1.275 to 1.35 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2135743.

INTENDED_EDIT: Change only the positive inference-time calibration multiplier; architecture, training, and ensemble decisions remain unchanged.

EVIDENCE: Successive scales of 1.10, 1.20, 1.25, and 1.275 preserved 9,253 correct while reducing cross-entropy to 0.2224249, 0.2160126, 0.2142010, and 0.2135743 respectively. Positive scaling preserves argmax, and the timeouts at intermediate scales provide no contrary calibration evidence because this scalar has negligible computational cost.

<<<<<<< SEARCH
        return 1.275 * ensemble_scores
=======
        return 1.35 * ensemble_scores
>>>>>>> REPLACE