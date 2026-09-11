MECHANISM: Finely bracketed terminal parameter EMA

HYPOTHESIS: An EMA update rate of 0.024 will exceed 9,328 correct predictions by reducing parameter lag slightly while avoiding the late-step sensitivity observed at 0.03 and 0.04.

INTENDED_EDIT: Increase the second-half parameter-EMA rate from 0.02 to 0.024 without changing training dynamics, BatchNorm handling, or runtime.

EVIDENCE: The 0.02 baseline scored 9,328, while 0.03 scored 9,325, 0.04 scored 9,326, and 0.015 scored 9,321; these results bracket a narrow optimum near the current setting and motivate a conservative interpolation.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.024)
>>>>>>> REPLACE