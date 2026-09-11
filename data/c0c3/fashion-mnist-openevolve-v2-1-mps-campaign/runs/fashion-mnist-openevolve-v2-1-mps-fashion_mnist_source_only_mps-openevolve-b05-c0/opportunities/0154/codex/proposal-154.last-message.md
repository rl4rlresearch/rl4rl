MECHANISM: Equivalent-decay dense EMA sampling

HYPOTHESIS: Updating the EMA every optimizer step with the four-step-equivalent decay rate will exceed 9,260 correct predictions by averaging the same temporal horizon without aliasing three of every four late-training iterates.

INTENDED_EDIT: Replace quarter-rate EMA updates at 0.03 interpolation with per-step updates at 0.007586, preserving the effective decay across each four-step interval.

EVIDENCE: The 9,260-correct baseline remains strongest, while terminal-weight interpolation produced no validation evidence and broader changes reduced correctness; refining the verified pure-EMA trajectory without changing its effective averaging horizon is the most conservative untested lever.

<<<<<<< SEARCH
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
=======
            elif completed_steps > ema_start:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.007586)
>>>>>>> REPLACE