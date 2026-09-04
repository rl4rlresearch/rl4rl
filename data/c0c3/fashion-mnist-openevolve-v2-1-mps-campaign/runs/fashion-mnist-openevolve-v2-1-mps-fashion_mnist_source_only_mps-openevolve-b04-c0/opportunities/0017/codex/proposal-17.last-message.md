MECHANISM: Further-shortened coherent EMA horizon

HYPOTHESIS: Reducing EMA decay from 0.98 to 0.97 will exceed 9,354 correct predictions by tracking the stronger late-training trajectory more responsively while retaining useful averaging.

INTENDED_EDIT: Increase both parameter and floating-buffer EMA update rates from 2% to 3%, preserving all other behavior.

EVIDENCE: Shortening coherent EMA decay from 0.99 to 0.98 improved validation_correct from 9,350 to 9,354, whereas lengthening it to 0.995 regressed to 9,336; continuing in the successful direction is the most informative next bracket.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.02)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.02)
=======
                    average.lerp_(parameter.detach(), 0.03)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.03)
>>>>>>> REPLACE