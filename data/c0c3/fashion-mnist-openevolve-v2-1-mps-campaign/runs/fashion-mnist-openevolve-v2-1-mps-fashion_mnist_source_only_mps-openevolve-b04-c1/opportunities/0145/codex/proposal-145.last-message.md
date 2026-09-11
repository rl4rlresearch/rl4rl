MECHANISM: Longer-horizon terminal parameter EMA

HYPOTHESIS: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise; the opposite change to 0.04 reduced performance to 9,326.

INTENDED_EDIT: Double the effective parameter-averaging horizon during the second half of training while leaving BatchNorm-buffer copying and all training dynamics unchanged.

EVIDENCE: Increasing the EMA rate to 0.04 scored 9,326 versus the 9,328 baseline, while averaging BatchNorm buffers scored 9,327; this motivates testing stronger parameter smoothing without altering normalization statistics.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.01)
>>>>>>> REPLACE