MECHANISM: Shorter-horizon coherent weight and BatchNorm averaging

HYPOTHESIS: Reducing the second-half EMA decay from 0.99 to 0.98 will exceed 9,350 correct predictions by tracking the better-trained late trajectory more responsively while retaining variance reduction.

INTENDED_EDIT: Increase both parameter and floating-buffer EMA update rates from 1% to 2%, preserving all other behavior.

EVIDENCE: The 0.99 coherent EMA supports the current 9,350-correct design, while lengthening its horizon with 0.995 regressed to 9,336; testing a shorter horizon is the most informative bracket of this validated mechanism.

<<<<<<< SEARCH
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.01)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.01)
=======
                for average, parameter in zip(
                    ema_parameters, model.parameters()
                ):
                    average.lerp_(parameter.detach(), 0.02)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.02)
>>>>>>> REPLACE