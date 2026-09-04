MECHANISM: Final-iterate EMA dose continuation

HYPOTHESIS: Increasing the forced terminal EMA interpolation from 6% to 9% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120091377.

INTENDED_EDIT: Use a 9% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.

EVIDENCE: Raising the terminal interpolation from 3% to 6% preserved 9,260 correct predictions and reduced cross-entropy from 0.2120130508 to 0.2120091377, motivating another runtime-neutral step in the same direction.

<<<<<<< SEARCH
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.06)
=======
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.09)
>>>>>>> REPLACE