MECHANISM: Boundary-refined BatchNorm-buffer EMA

HYPOTHESIS: A 3.125% buffer EMA update will retain 9,359 correct predictions while lowering cross-entropy below 0.193865 by moving toward the better-calibrated 3.25% rate without crossing its observed accuracy boundary.

INTENDED_EDIT: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.125%.

EVIDENCE: Buffer updates of 2.5%, 2.75%, and 3% all produced 9,359 correct while cross-entropy improved monotonically; 3.25% further improved cross-entropy but lost one correct prediction, making their midpoint the closest untested boundary refinement.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03125)
>>>>>>> REPLACE