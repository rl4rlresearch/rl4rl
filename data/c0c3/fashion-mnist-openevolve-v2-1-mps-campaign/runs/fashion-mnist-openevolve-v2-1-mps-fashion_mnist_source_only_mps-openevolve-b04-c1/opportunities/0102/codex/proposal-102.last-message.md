MECHANISM: Longer-horizon parameter EMA

HYPOTHESIS: Reducing the EMA update rate from 0.02 to 0.01 will exceed 9,328 correct predictions by averaging over more late-training updates and suppressing trajectory variance without changing optimization or runtime materially.

INTENDED_EDIT: Double the effective parameter-averaging horizon during the second half of training.

EVIDENCE: Learning-rate-floor changes in both directions failed to improve the 9,328-correct baseline, while the prior runtime-neutral 0.01 EMA attempt timed out and produced no contrary accuracy evidence, leaving this stabilization hypothesis unresolved.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
=======
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.01)
                    else:
>>>>>>> REPLACE