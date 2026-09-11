MECHANISM: Symmetric EMA horizon refinement

HYPOTHESIS: Decreasing the parameter-EMA interpolation rate from 0.020 to 0.018 will exceed 9,328 correct predictions by modestly suppressing noisy late updates without approaching the overly long horizon that underperformed at 0.010.

INTENDED_EDIT: Change only the parameter-EMA interpolation rate from 0.020 to 0.018.

EVIDENCE: Increasing the rate to 0.022 tied the 0.020 baseline at 9,328 correct but worsened cross-entropy, while 0.010 and 0.030 reached only 9,322 and 9,325; a symmetric downward refinement is the most targeted remaining probe near the observed optimum.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.018)
>>>>>>> REPLACE