MECHANISM: Evidence-fitted low-consensus confidence calibration

HYPOTHESIS: Setting the below-average-agreement bonus to 0.012 will preserve all 9,360 correct predictions while lowering validation cross-entropy below 0.18569551315307617.

INTENDED_EDIT: Reduce only the centered-hinge confidence coefficient from 0.02 to 0.012.

EVIDENCE: A negative low-agreement adjustment worsened cross-entropy to 0.1856981330871582, while a 0.02 bonus improved it to 0.18569551315307617 without changing predictions; a quadratic fit across the negative, zero, and positive results estimates the optimum near 0.012.

<<<<<<< SEARCH
            0.31 * agreement + 0.02 * F.relu(-agreement)
=======
            0.31 * agreement + 0.012 * F.relu(-agreement)
>>>>>>> REPLACE