MECHANISM: Moderately longer-horizon terminal parameter EMA

HYPOTHESIS: Reducing the parameter-EMA update rate from 0.02 to 0.015 will exceed 9,328 correct predictions by suppressing late optimizer noise without the stronger lag of the unresolved 0.01 setting.

INTENDED_EDIT: Increase the effective second-half parameter-averaging horizon from roughly 50 to 67 optimizer steps while continuing to copy BatchNorm buffers directly.

EVIDENCE: Increasing the EMA rate to 0.04 reduced correctness from 9,328 to 9,326, indicating that a shorter averaging horizon is harmful; 0.015 is a conservative directional test because the more aggressive 0.01 setting was never successfully verified.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.015)
>>>>>>> REPLACE