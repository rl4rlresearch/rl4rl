MECHANISM: Further-slowed BatchNorm-buffer EMA

HYPOTHESIS: Keeping parameter EMA at 0.96 decay while increasing buffer EMA decay from 0.97 to 0.98 will exceed 9,359 correct predictions—or tie with lower cross-entropy—by further reducing noise in BatchNorm running statistics without changing the best-tested parameter trajectory.

INTENDED_EDIT: Preserve the 4% parameter EMA update and reduce only the floating-buffer EMA update from 3% to 2%.

EVIDENCE: Reducing the buffer update from the coherent 4% rate to 3% improved validation_correct from 9,358 to 9,359, whereas increasing it to 6.25% regressed to 9,355; another measured reduction directly tests the successful direction.

<<<<<<< SEARCH
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.02)
>>>>>>> REPLACE