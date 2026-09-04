MECHANISM: Shorter-horizon coherent EMA

HYPOTHESIS: Reducing EMA decay from 0.97 to 0.96 will exceed 9,356 correct predictions by following the stronger late-training trajectory more responsively while preserving useful weight and BatchNorm averaging.

INTENDED_EDIT: Increase both parameter and floating-buffer EMA update rates from 3% to 4%, preserving all other training and evaluation behavior.

EVIDENCE: Successively shortening coherent EMA decay from 0.99 to 0.98 to 0.97 increased validation_correct from 9,350 to 9,354 to 9,356, so continuing one measured step in the validated direction is the most informative next bracket.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
>>>>>>> REPLACE