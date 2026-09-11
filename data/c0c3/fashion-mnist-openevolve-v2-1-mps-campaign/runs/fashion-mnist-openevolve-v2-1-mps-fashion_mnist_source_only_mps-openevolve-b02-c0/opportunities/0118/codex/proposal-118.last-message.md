MECHANISM: Three-point quadratic post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3717 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125840324.

INTENDED_EDIT: Increase only the positive inference-time ensemble calibration multiplier from 1.3706 to 1.3717.

EVIDENCE: Cross-entropy fell from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698 and 0.2125840324 at 1.3706; a quadratic fit to these verified points places the minimum near 1.3717, while positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.3706 * ensemble_scores
=======
        return 1.3717 * ensemble_scores
>>>>>>> REPLACE