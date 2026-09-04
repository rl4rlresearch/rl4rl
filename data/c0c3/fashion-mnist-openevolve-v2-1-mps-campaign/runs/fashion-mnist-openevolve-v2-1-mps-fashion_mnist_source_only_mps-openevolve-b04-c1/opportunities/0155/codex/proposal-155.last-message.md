MECHANISM: Intermediate-horizon terminal parameter EMA

HYPOTHESIS: Increasing the EMA update rate from 0.02 to 0.03 will exceed 9,328 correct predictions by reducing parameter lag while avoiding the excess late-step sensitivity observed at 0.04.

INTENDED_EDIT: Shorten the second-half EMA horizon from roughly 50 to 33 optimizer steps without changing training dynamics or BatchNorm-buffer handling.

EVIDENCE: EMA rates bracket the current setting: 0.015 scored 9,321, 0.02 scored 9,328, and 0.04 scored 9,326. This motivates testing 0.03 between the two strongest settings.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.03)
>>>>>>> REPLACE