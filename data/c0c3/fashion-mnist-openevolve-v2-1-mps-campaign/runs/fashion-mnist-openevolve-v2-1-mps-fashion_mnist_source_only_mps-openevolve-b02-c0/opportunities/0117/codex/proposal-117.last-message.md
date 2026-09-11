MECHANISM: Conservative post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3706 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.

INTENDED_EDIT: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3706.

EVIDENCE: Cross-entropy improved through 1.3698, and the estimated convex minimum lies near 1.3715. Testing the midpoint toward that estimate should retain most of the predicted gain; positive scaling preserves argmax predictions, while the 1.3715 attempts timed out without contradictory score evidence.

<<<<<<< SEARCH
        return 1.3698 * ensemble_scores
=======
        return 1.3706 * ensemble_scores
>>>>>>> REPLACE