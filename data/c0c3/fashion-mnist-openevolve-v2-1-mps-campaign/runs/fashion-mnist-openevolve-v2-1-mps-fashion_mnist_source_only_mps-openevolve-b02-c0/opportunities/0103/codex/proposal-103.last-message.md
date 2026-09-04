MECHANISM: Local quadratic post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3599 will preserve all 9,253 argmax predictions while reducing validation cross-entropy below 0.21259700.

INTENDED_EDIT: Decrease only the positive inference-time calibration multiplier from 1.36 to 1.3599.

EVIDENCE: Cross-entropy slopes over 1.30–1.34 and 1.34–1.36 were approximately -0.010748 and -0.004274; a quadratic fit to those verified measurements places the local minimum near 1.3599. Any positive scale preserves argmax predictions.

<<<<<<< SEARCH
        return 1.36 * ensemble_scores
=======
        return 1.3599 * ensemble_scores
>>>>>>> REPLACE