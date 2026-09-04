MECHANISM: Horizon-aligned BatchNorm-statistics EMA

HYPOTHESIS: Averaging floating-point BatchNorm buffers at 0.025 will exceed 9,328 correct predictions by matching their combined tracking lag to the successful 0.02 parameter-EMA horizon.

INTENDED_EDIT: Apply a 0.025 EMA to floating-point model buffers during parameter averaging while continuing to copy integer buffers directly.

EVIDENCE: Averaging BatchNorm buffers at the parameter rate reached 9,327 correct, suggesting normalization alignment is useful but slightly over-lagged; accounting for BatchNorm’s existing momentum makes 0.025 a closer horizon match.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    elif torch.is_floating_point(averaged):
                        averaged.lerp_(value.detach(), 0.025)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE