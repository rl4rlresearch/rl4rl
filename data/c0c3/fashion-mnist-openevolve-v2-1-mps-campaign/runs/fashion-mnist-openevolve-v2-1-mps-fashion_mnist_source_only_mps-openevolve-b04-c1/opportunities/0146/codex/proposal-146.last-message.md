MECHANISM: Longer-horizon terminal parameter EMA

HYPOTHESIS: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise; increasing the rate to 0.04 reduced performance to 9,326.

INTENDED_EDIT: Double the effective parameter-averaging horizon during the second half of training while retaining terminal BatchNorm buffers and all other training dynamics.

EVIDENCE: The verified 0.04 EMA scored 9,326 versus the 9,328 baseline, providing directional evidence that shorter averaging is harmful; the corresponding longer-horizon 0.01 setting remains unmeasured because prior verification attempts did not complete.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.01)
>>>>>>> REPLACE