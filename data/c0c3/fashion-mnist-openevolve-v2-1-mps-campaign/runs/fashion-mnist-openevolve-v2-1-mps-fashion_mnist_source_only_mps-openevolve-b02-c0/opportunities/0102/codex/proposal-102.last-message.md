MECHANISM: Quadratic-fit post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores from 1.36 to 1.3698 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21259700.

INTENDED_EDIT: Change only the positive inference-time ensemble calibration multiplier from 1.36 to 1.3698.

EVIDENCE: Cross-entropy fell from 0.21311241 at 1.30 to 0.21268247 at 1.34 and 0.21259700 at 1.36, with all 9,253 argmax predictions preserved. A quadratic fit to these measurements places the minimum near 1.3698; the timed-out 1.37 verification supplied no contradictory score evidence.

<<<<<<< SEARCH
        return 1.36 * ensemble_scores
=======
        return 1.3698 * ensemble_scores
>>>>>>> REPLACE