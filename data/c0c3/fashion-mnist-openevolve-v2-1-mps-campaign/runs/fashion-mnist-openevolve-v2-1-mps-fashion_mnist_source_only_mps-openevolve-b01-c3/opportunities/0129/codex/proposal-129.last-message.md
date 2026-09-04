MECHANISM: Stronger view-agreement-conditioned logit calibration

HYPOTHESIS: A 30% disagreement adjustment will preserve exactly 9,287 correct predictions through positive per-image scaling while reducing validation cross-entropy below 0.20670405883789061.

INTENDED_EDIT: Double the agreement-conditioned calibration strength from 15% to 30%, leaving training and pooled predictions unchanged.

EVIDENCE: The verified 15% adjustment preserved 9,287 correct and improved cross-entropy; both 30% attempts only timed out, so this accuracy-invariant line-search point remains unresolved.

<<<<<<< SEARCH
        calibration = 1.22775 * (0.85 + 0.15 * view_agreement)
=======
        calibration = 1.22775 * (0.70 + 0.30 * view_agreement)
>>>>>>> REPLACE