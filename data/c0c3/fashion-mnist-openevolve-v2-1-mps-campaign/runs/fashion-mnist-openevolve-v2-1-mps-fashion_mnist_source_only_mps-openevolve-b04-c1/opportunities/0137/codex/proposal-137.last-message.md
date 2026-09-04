MECHANISM: Longer-horizon terminal parameter EMA

HYPOTHESIS: Reducing the parameter-EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by suppressing late optimizer noise while retaining the verified training trajectory and current BatchNorm-buffer handling.

INTENDED_EDIT: Double the effective parameter-averaging horizon during the second half of training without changing architecture, loss, learning-rate schedule, runtime, or buffer copying.

EVIDENCE: The 0.02 parameter EMA produced the 9,328 baseline, while architecture, augmentation, and late-loss changes failed to improve it; the unresolved lower-learning-rate-floor experiments specifically motivate testing runtime-neutral late-update stabilization through stronger averaging.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
=======
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.01)
                    else:
>>>>>>> REPLACE