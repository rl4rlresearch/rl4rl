MECHANISM: Shorter-horizon sparse EMA

HYPOTHESIS: Increasing the sparse EMA interpolation rate from 0.03 to 0.04 will exceed 9,260 correct predictions by retaining more late decision-boundary refinement without altering the training trajectory or adding per-step overhead.

INTENDED_EDIT: Keep quarter-rate EMA updates but shorten their averaging horizon by increasing the interpolation rate to 0.04.

EVIDENCE: The immediate-cosine baseline remains strongest, while warmup reduced correctness and terminal-weight interpolation produced no validation evidence; the per-step EMA experiment also timed out, motivating a runtime-neutral test of later-weight emphasis within the verified sparse EMA procedure.

<<<<<<< SEARCH
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.03)
=======
            elif (completed_steps - ema_start) % 4 == 0:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.04)
>>>>>>> REPLACE