MECHANISM: EMA-aligned BatchNorm buffer averaging

HYPOTHESIS: Averaging BatchNorm running means and variances on the same 0.02 EMA horizon as learned parameters will exceed 9,328 correct predictions by evaluating the averaged model with normalization statistics from the same late-training trajectory.

INTENDED_EDIT: EMA-average floating-point BatchNorm running statistics instead of copying only their final values; continue copying integer counters directly.

EVIDENCE: Live paired-view BatchNorm tied the best 9,328 correct predictions, while setting BatchNorm momentum to 0.02 reached 9,326, showing normalization-state alignment is influential but unresolved; directly averaging the buffers with the parameters tests that alignment without changing training computation or learned parameters.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    elif value.is_floating_point() and (
                        name.endswith("running_mean")
                        or name.endswith("running_var")
                    ):
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE