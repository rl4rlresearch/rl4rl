MECHANISM: Longer-horizon terminal parameter EMA

HYPOTHESIS: Reducing the EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise during the final half of training.

INTENDED_EDIT: Double the effective parameter-averaging horizon while retaining terminal BatchNorm buffers and all training dynamics.

EVIDENCE: Increasing the EMA rate to 0.04 reduced correctness to 9,326 versus the 9,328 baseline, providing directional evidence that shorter averaging is harmful; the complementary 0.01 rate remains unmeasured because prior verification attempts did not complete.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.01)
>>>>>>> REPLACE