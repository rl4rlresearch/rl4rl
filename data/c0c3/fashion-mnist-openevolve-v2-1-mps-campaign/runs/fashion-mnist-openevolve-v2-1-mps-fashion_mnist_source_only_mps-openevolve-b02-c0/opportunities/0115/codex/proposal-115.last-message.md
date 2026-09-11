MECHANISM: Secant-curvature post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3715 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.2125842613.

INTENDED_EDIT: Increase only the positive inference-time ensemble calibration multiplier from 1.3698 to 1.3715.

EVIDENCE: Cross-entropy improved from 0.2125948353 at 1.3610 to 0.2125842613 at 1.3698, and the estimated convex minimum is near 1.3715. Positive scaling preserves argmax predictions; the prior 1.3715 timeout supplied no contradictory score evidence.

<<<<<<< SEARCH
        return 1.3698 * ensemble_scores
=======
        return 1.3715 * ensemble_scores
>>>>>>> REPLACE