MECHANISM: Upper-side BatchNorm-buffer EMA midpoint refinement

HYPOTHESIS: A 3.5% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by retaining most of the 3% rate’s accuracy gain while moving toward the better-calibrated 4% rate.

INTENDED_EDIT: Keep parameter EMA at 4% and increase only the floating-buffer EMA update from 3% to 3.5%.

EVIDENCE: The 3% buffer update achieved the best result at 9,359 correct; 2% and 4% each achieved 9,358, while 4% had lower cross-entropy than 3%, motivating the untested midpoint between 3% and 4%.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.035)
>>>>>>> REPLACE