MECHANISM: Lower-side BatchNorm-buffer EMA midpoint refinement

HYPOTHESIS: A 2.5% buffer EMA update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by balancing the 3% rate’s accuracy against the stronger smoothing of the 2% rate while preserving the best-tested 4% parameter EMA.

INTENDED_EDIT: Keep parameter EMA at 4% and reduce only the floating-buffer EMA update from 3% to 2.5%.

EVIDENCE: The 3% buffer update achieved the best result at 9,359 correct, while 2% achieved 9,358; testing their untried midpoint is the most informative refinement after the 3.5% verification failed to finish.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.025)
>>>>>>> REPLACE