MECHANISM: Faster EMA for normalization-state alignment

HYPOTHESIS: Increasing the parameter EMA interpolation from 0.02 to 0.04 will exceed 9,328 correct predictions by reducing the age mismatch between averaged weights and the final BatchNorm statistics.

INTENDED_EDIT: Shorten the parameter EMA timescale from roughly 50 steps to 25 steps without adding per-step state processing or changing the successful architecture and regularization schedules.

EVIDENCE: EMA-aligned regularization reached 9,328 correct, while attempts to average BatchNorm buffers or slow their momentum timed out; a moderately faster parameter EMA tests the same state-consistency mechanism at negligible runtime cost.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.04)
>>>>>>> REPLACE