MECHANISM: Quadratic-fit post-ensemble temperature calibration

HYPOTHESIS: Increasing the positive ensemble scale from 1.36 to 1.37 will preserve exactly 9,253 correct predictions while reducing validation cross-entropy below 0.21259700.

INTENDED_EDIT: Change only the inference-time ensemble calibration multiplier from 1.36 to 1.37.

EVIDENCE: Scaling from 1.30 to 1.34 reduced cross-entropy by 0.00042994, and scaling from 1.34 to 1.36 reduced it by another 0.00008547 while preserving all 9,253 predictions. These finite differences place the local convex optimum near 1.370; positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.36 * ensemble_scores
=======
        return 1.37 * ensemble_scores
>>>>>>> REPLACE