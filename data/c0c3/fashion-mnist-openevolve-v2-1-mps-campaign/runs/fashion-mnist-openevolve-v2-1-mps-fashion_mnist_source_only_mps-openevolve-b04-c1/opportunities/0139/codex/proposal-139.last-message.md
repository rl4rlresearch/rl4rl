MECHANISM: EMA-aligned normalization buffers

HYPOTHESIS: Averaging floating-point BatchNorm buffers at the same 0.02 rate as parameters will exceed 9,328 correct predictions by reducing the normalization mismatch between terminal running statistics and EMA-averaged weights.

INTENDED_EDIT: During second-half parameter EMA, also EMA-average BatchNorm running means and variances while continuing to copy integer tracking buffers directly.

EVIDENCE: The 9,328 baseline copies short-horizon BatchNorm statistics into its longer-horizon EMA model; the unverified low-momentum BatchNorm proposal identified this mismatch, and buffer averaging tests it directly without altering training dynamics or runtime materially.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if (
                        name in optimizer.ema_parameter_names
                        or value.is_floating_point()
                    ):
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE