MECHANISM: Intermediate final-iterate EMA interpolation

HYPOTHESIS: Raising terminal EMA interpolation from 6% to 7.5% will retain 9,260 correct predictions while reducing validation cross-entropy below 0.2120091377.

INTENDED_EDIT: Increase only the forced final optimizer-iterate EMA interpolation to 7.5%, preserving regular 3% quarter-rate EMA updates and all other behavior.

EVIDENCE: Increasing terminal interpolation from 3% to 6% preserved 9,260 correct and improved cross-entropy from 0.2120130508 to 0.2120091377. The prior 7.5% verification timed out, providing no validation evidence against this runtime-neutral continuation.

<<<<<<< SEARCH
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.06)
=======
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.075)
>>>>>>> REPLACE