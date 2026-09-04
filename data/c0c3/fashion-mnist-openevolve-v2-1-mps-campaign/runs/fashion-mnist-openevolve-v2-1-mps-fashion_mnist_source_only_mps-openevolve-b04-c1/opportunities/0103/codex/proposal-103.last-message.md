MECHANISM: Shorter-horizon parameter EMA

HYPOTHESIS: Increasing the EMA update rate from 0.02 to 0.03 will exceed 9,328 correct predictions by tracking the improving late-training trajectory more responsively while still averaging roughly 33 optimizer updates.

INTENDED_EDIT: Increase only the parameter-EMA interpolation rate from 0.02 to 0.03.

EVIDENCE: Extending the EMA horizon with a 0.01 update rate reduced validation correct from 9,328 to 9,322, directly motivating a runtime-neutral test in the opposite direction.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.03)
>>>>>>> REPLACE