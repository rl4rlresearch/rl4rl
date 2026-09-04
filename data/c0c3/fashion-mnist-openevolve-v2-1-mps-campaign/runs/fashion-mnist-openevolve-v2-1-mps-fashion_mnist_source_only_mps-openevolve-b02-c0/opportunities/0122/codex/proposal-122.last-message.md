MECHANISM: Quadratic-vertex post-ensemble temperature calibration

HYPOTHESIS: Scaling ensemble scores by 1.3715955 will preserve all 9,253 correct predictions while reducing validation cross-entropy below 0.2125839283.

INTENDED_EDIT: Refine only the positive inference-time ensemble calibration multiplier from 1.3716 to 1.3715955.

EVIDENCE: Verified cross-entropies at scales 1.3706, 1.3716, and 1.3717 place the local convex minimum near 1.3715955; the prior attempt at this value timed out without contradictory score evidence, and positive scaling preserves argmax predictions.

<<<<<<< SEARCH
        return 1.3716 * ensemble_scores
=======
        return 1.3715955 * ensemble_scores
>>>>>>> REPLACE