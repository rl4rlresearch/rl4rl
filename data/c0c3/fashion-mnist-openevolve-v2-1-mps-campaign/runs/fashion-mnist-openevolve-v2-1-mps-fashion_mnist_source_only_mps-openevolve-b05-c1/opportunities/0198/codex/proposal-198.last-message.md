MECHANISM: Incremental high-consensus confidence calibration

HYPOTHESIS: Increasing the above-average-agreement bonus from 0.028 to 0.042 will preserve all 9,360 argmax predictions while lowering validation cross-entropy below 0.1856942943572998.

INTENDED_EDIT: Raise only the positive-agreement centered-hinge coefficient, increasing the high-agreement confidence slope from 0.338 to 0.352 while leaving low-agreement calibration unchanged.

EVIDENCE: Successive 0.014 increases reduced cross-entropy from 0.18569544525146484 to 0.18569485626220703 and then 0.1856942943572998 with identical predictions. The prior 0.042 attempt timed out, providing no subject-level evidence against this continuation.

<<<<<<< SEARCH
            + 0.028 * F.relu(agreement)
=======
            + 0.042 * F.relu(agreement)
>>>>>>> REPLACE