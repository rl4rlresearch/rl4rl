MECHANISM: Refined finite-difference temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3716 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839294.

INTENDED_EDIT: Decrease only the positive inference-time ensemble calibration multiplier from 1.3717 to 1.3716.

EVIDENCE: Cross-entropy improved through scales 1.3698, 1.3706, and 1.3717; the slopes across the last two intervals imply a convex minimum near 1.3716. Positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.3717 * ensemble_scores
=======
        return 1.3716 * ensemble_scores
>>>>>>> REPLACE