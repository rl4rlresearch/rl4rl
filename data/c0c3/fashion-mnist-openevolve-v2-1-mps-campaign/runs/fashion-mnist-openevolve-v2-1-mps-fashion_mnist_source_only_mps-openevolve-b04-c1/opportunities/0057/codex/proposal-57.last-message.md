MECHANISM: State-consistent EMA of BatchNorm statistics

HYPOTHESIS: Averaging floating-point BatchNorm buffers with the EMA parameters will exceed 9,328 correct predictions by reducing normalization noise and the mismatch between averaged weights and final-model statistics.

INTENDED_EDIT: Extend EMA interpolation to every floating-point model state while continuing to copy integer tracking buffers directly.

EVIDENCE: EMA-aligned dropout and label-smoothing annealing reached 9,328 correct; the previous state-consistent EMA verification timed out, leaving this low-cost correction unresolved rather than disproven.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if value.is_floating_point():
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE