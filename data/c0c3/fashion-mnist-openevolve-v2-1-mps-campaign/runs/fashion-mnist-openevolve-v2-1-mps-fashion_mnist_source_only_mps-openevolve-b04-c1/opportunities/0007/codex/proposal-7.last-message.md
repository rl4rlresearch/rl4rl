MECHANISM: Longer-horizon late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA horizon from roughly 100 to 200 optimizer steps will exceed 9,237 correct predictions by further suppressing small-batch parameter and BatchNorm-state variance while retaining emphasis on the converged trajectory.

INTENDED_EDIT: Reduce the EMA interpolation rate from 0.01 to 0.005, preserving the proven architecture, schedule, augmentation, and averaging start point.

EVIDENCE: Late-training EMA improved the same small-batch model from 9,215 to 9,237 correct; extending that successful averaging mechanism is the most direct test of whether additional trajectory smoothing yields further gains.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.01)
=======
                        averaged.lerp_(value.detach(), 0.005)
>>>>>>> REPLACE