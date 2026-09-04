MECHANISM: Evidence-fitted low-consensus calibration refinement

HYPOTHESIS: Setting the below-average-agreement bonus to 0.014 will preserve all 9,360 correct predictions while lowering validation cross-entropy below 0.1856954532623291.

INTENDED_EDIT: Increase only the centered-hinge confidence coefficient from 0.012 to 0.014.

EVIDENCE: At the same 0.31 agreement coefficient, bonuses of 0, 0.012, and 0.02 yielded cross-entropies of 0.18569574165344238, 0.1856954532623291, and 0.18569551315307617 without changing predictions; a quadratic fit to these verified points places the minimum near 0.0136.

<<<<<<< SEARCH
            0.31 * agreement + 0.012 * F.relu(-agreement)
=======
            0.31 * agreement + 0.014 * F.relu(-agreement)
>>>>>>> REPLACE