MECHANISM: Further-shortened coherent EMA horizon

HYPOTHESIS: Reducing EMA decay from 0.96 to 0.95 will exceed 9,358 correct predictions by tracking the stronger late-training trajectory more responsively while retaining useful weight and BatchNorm averaging.

INTENDED_EDIT: Increase both parameter and floating-buffer EMA update rates from 4% to 5%, preserving all other training and evaluation behavior.

EVIDENCE: Successively shortening coherent EMA decay from 0.99 to 0.98, 0.97, and 0.96 increased validation_correct from 9,350 to 9,354, 9,356, and 9,358; continuing one measured step is the most informative test of this monotonic trend.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.04)
=======
                    average.lerp_(parameter.detach(), 0.05)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.05)
>>>>>>> REPLACE