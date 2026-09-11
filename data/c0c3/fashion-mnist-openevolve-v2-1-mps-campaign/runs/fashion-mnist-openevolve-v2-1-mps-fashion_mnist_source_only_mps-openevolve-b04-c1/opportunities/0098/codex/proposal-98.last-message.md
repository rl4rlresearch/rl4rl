MECHANISM: Longer-horizon parameter EMA

HYPOTHESIS: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late-update variance while retaining enough responsiveness to the cosine-decayed trajectory.

INTENDED_EDIT: Double the effective averaging horizon of learned parameters without changing training computation, normalization buffers, or the optimizer schedule.

EVIDENCE: Changing the learning-rate floor in either tested direction reduced accuracy, while BatchNorm momentum changes also failed to improve the 9,328-correct baseline; this motivates stabilizing the successful trajectory through parameter averaging rather than altering its updates or normalization.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.01)
>>>>>>> REPLACE