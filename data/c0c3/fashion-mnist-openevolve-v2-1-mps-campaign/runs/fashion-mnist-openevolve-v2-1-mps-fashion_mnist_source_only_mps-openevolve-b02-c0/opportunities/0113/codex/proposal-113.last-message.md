MECHANISM: Quadratic-fit post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3698 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125948353.

INTENDED_EDIT: Increase only the positive inference-time ensemble calibration multiplier from 1.3610 to 1.3698.

EVIDENCE: Verified cross-entropy improvements through 1.3610 establish that the improving direction remains upward, while the earlier finite-difference quadratic fit places the estimated minimum near 1.3698; previous attempts at that scale timed out without contradictory score evidence, and positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.3610 * ensemble_scores
=======
        return 1.3698 * ensemble_scores
>>>>>>> REPLACE