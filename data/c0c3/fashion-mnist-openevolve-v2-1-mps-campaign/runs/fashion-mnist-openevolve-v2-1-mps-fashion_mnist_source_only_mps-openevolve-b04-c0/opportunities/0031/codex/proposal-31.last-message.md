MECHANISM: Decoupled weight-and-buffer EMA refinement

HYPOTHESIS: Reducing the parameter EMA update to 3.75% while retaining the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by combining the stronger parameter smoothing associated with 0.193577 cross-entropy and the buffer horizon responsible for the current accuracy gain.

INTENDED_EDIT: Change only the parameter EMA update rate from 4% to 3.75%; preserve the 3% floating-buffer update and all other behavior.

EVIDENCE: The coherent 3.75% EMA produced the lowest observed cross-entropy, 0.193577, but lost one correct prediction; independently slowing buffer updates to 3% raised the best result to 9,359 correct, so decoupling these rates directly tests whether their benefits combine.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.0375)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
>>>>>>> REPLACE