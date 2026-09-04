MECHANISM: Final-iterate EMA inclusion

HYPOTHESIS: Incorporating the terminal optimizer iterate into the sparse EMA with a conservative 3% interpolation will exceed 9,260 correct predictions by capturing late decision-boundary refinement without materially reducing EMA smoothing.

INTENDED_EDIT: Force one final EMA update when training completes, while preserving the architecture, optimizer, schedule, BatchNorm mixture, and evaluation calibration.

EVIDENCE: The verified sparse EMA reaches 9,260 correct but its four-step update phase omits the final three iterates; a larger 10% terminal interpolation timed out without validation evidence, motivating this smaller runtime-neutral test.

<<<<<<< SEARCH
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
=======
            elif (
                (completed_steps - ema_start) % 4 == 0
                or completed_steps >= total_steps
            ):
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
>>>>>>> REPLACE