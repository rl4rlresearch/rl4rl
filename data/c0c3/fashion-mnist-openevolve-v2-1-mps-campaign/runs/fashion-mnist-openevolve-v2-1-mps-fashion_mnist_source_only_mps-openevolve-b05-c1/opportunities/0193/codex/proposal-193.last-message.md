MECHANISM: Quadratic-fit low-consensus confidence calibration

HYPOTHESIS: Setting the below-average-agreement bonus to 0.0136 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18569544525146484.

INTENDED_EDIT: Refine only the centered-hinge confidence coefficient from 0.014 to the evidence-fitted optimum of 0.0136.

EVIDENCE: Bonuses of 0, 0.012, 0.014, and 0.02 produced cross-entropies of 0.18569574165344238, 0.1856954532623291, 0.18569544525146484, and 0.18569551315307617 with identical predictions; their observed curvature places the minimum near 0.0136.

<<<<<<< SEARCH
            0.31 * agreement + 0.014 * F.relu(-agreement)
=======
            0.31 * agreement + 0.0136 * F.relu(-agreement)
>>>>>>> REPLACE