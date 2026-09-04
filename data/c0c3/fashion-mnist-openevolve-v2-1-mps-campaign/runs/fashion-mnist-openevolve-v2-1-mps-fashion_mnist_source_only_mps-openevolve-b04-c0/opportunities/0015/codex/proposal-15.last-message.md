MECHANISM: Longer-horizon exponential weight and BatchNorm averaging

HYPOTHESIS: Increasing the second-half EMA decay from 0.99 to 0.995 will exceed 9,350 correct predictions by reducing late-training variance while retaining more of the stable trajectory.

INTENDED_EDIT: Slow both parameter and floating-buffer EMA updates from 1% to 0.5%, preserving the architecture, loss, schedule, exposure, and coherent BatchNorm averaging.

EVIDENCE: Parameter EMA improved validation_correct from 9,322 to 9,323, and extending it to BatchNorm buffers improved it further to 9,335; with auxiliary-loss tuning now bracketed around the 9,350 design, EMA horizon is the strongest validated mechanism left for a focused refinement.

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
                    average.lerp_(parameter.detach(), 0.005)
                for average, buffer in zip(
                    ema_buffers, floating_buffers
                ):
                    average.lerp_(buffer.detach(), 0.005)
>>>>>>> REPLACE