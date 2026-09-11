MECHANISM: Slower BatchNorm-buffer EMA

HYPOTHESIS: Keeping parameter EMA at 0.96 decay while increasing buffer decay to 0.97 will exceed 9,358 correct predictions—or tie with lower cross-entropy—by smoothing BatchNorm running statistics more strongly without altering the best-tested parameter trajectory.

INTENDED_EDIT: Preserve the 4% parameter EMA update and reduce only the floating-buffer update from 4% to 3%.

EVIDENCE: The coherent 4% EMA achieved 9,358 correct, while increasing only the buffer update to 6.25% regressed to 9,355; testing a modest decrease to 3% directly probes the opposite side of the buffer-rate optimum.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
>>>>>>> REPLACE