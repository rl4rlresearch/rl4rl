MECHANISM: EMA-aligned BatchNorm statistics

HYPOTHESIS: Averaging BatchNorm running statistics at the proven 0.02 parameter-EMA rate will exceed 9,328 correct predictions by reducing the mismatch between smoothed parameters and terminal-model normalization statistics.

INTENDED_EDIT: Apply EMA to floating-point buffers such as BatchNorm running means and variances while continuing to copy integer counters directly.

EVIDENCE: The 0.02 parameter EMA scored 9,328, outperforming rates of 0.015, 0.024, 0.03, and 0.04, but its BatchNorm buffers currently come from the terminal model rather than the same successful averaging horizon.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    elif value.is_floating_point():
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE