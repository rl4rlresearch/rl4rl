MECHANISM: Incremental high-consensus confidence calibration

HYPOTHESIS: Increasing the above-average-agreement bonus from 0.028 to 0.042 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.1856942943572998.

INTENDED_EDIT: Increase only the positive-agreement centered-hinge coefficient, raising the high-agreement confidence slope from 0.338 to 0.352 while leaving the fitted low-agreement slope unchanged.

EVIDENCE: Successive 0.014 increases lowered cross-entropy from 0.18569544525146484 to 0.18569485626220703 and then 0.1856942943572998 with identical predictions; the second gain remained nearly as large as the first, supporting another equal-sized step.

<<<<<<< SEARCH
            + 0.028 * F.relu(agreement)
=======
            + 0.042 * F.relu(agreement)
>>>>>>> REPLACE