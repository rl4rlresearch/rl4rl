MECHANISM: Layerwise terminal EMA horizons

HYPOTHESIS: Using a 0.04 EMA update rate for the classifier while retaining 0.02 for feature parameters will exceed 9,328 correct predictions by tracking the rapidly changing dropout- and smoothing-free terminal classifier without exposing the backbone to the weaker global 0.04 averaging regime.

INTENDED_EDIT: Shorten only the classifier’s EMA horizon from roughly 50 to 25 optimizer steps during second-half averaging.

EVIDENCE: Global EMA rates of 0.024, 0.03, and 0.04 scored 9,327, 9,325, and 9,326 versus 9,328 at 0.02; because dropout and label smoothing anneal specifically during this averaging window, the close 0.04 result motivates isolating its faster tracking to the classifier while preserving the proven backbone horizon.

<<<<<<< SEARCH
                    if name in optimizer.ema_parameter_names:
                        averaged.lerp_(value.detach(), 0.02)
                    else:
                        averaged.copy_(value)
=======
                    if name in optimizer.ema_parameter_names:
                        ema_rate = (
                            0.04
                            if name.startswith("classifier.")
                            else 0.02
                        )
                        averaged.lerp_(value.detach(), ema_rate)
                    else:
                        averaged.copy_(value)
>>>>>>> REPLACE