MECHANISM: Fine-grained EMA horizon tuning

HYPOTHESIS: Increasing the parameter-EMA interpolation rate from 0.020 to 0.022 will exceed 9,328 correct predictions by modestly favoring later training states without the shorter-horizon degradation observed at 0.030.

INTENDED_EDIT: Change only the parameter-EMA interpolation rate from 0.020 to 0.022.

EVIDENCE: EMA rates of 0.010 and 0.030 produced 9,322 and 9,325 correct versus 9,328 at 0.020; these opposite-side probes place the best observed horizon near 0.020 and motivate a small upward refinement rather than another structural or runtime-heavy change.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.022)
>>>>>>> REPLACE