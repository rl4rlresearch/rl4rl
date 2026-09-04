MECHANISM: Final-iterate EMA dose refinement

HYPOTHESIS: Increasing only the forced terminal EMA interpolation from 3% to 6% will preserve 9,260 correct predictions while lowering validation cross-entropy below 0.2120130508.

INTENDED_EDIT: Use a 6% EMA interpolation for the final optimizer iterate while retaining 3% for regular quarter-rate EMA updates.

EVIDENCE: The newly verified 3% final-iterate inclusion improved the objective at unchanged accuracy, while the earlier 10% experiment timed out without validation evidence; 6% is a runtime-neutral intermediate test of the observed benefit.

<<<<<<< SEARCH
            elif (
                (completed_steps - ema_start) % 4 == 0
                or completed_steps >= total_steps
            ):
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
=======
            elif completed_steps >= total_steps:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.06)
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
>>>>>>> REPLACE