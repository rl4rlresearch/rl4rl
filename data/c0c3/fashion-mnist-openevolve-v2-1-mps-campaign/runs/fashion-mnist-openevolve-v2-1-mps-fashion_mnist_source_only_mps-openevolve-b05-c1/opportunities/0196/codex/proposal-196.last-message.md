MECHANISM: Incremental high-consensus confidence calibration

HYPOTHESIS: Increasing the above-average-agreement bonus from 0.014 to 0.028 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.18569485626220703.

INTENDED_EDIT: Increase only the positive-agreement centered-hinge coefficient, raising the high-agreement confidence slope from 0.324 to 0.338 while leaving the fitted low-agreement slope unchanged.

EVIDENCE: Adding the first 0.014 positive-agreement bonus lowered cross-entropy from 0.18569544525146484 to 0.18569485626220703 with identical predictions, so an equal-sized continuation is the most direct test of whether the high-consensus optimum lies higher.

<<<<<<< SEARCH
            + 0.014 * F.relu(agreement)
=======
            + 0.028 * F.relu(agreement)
>>>>>>> REPLACE