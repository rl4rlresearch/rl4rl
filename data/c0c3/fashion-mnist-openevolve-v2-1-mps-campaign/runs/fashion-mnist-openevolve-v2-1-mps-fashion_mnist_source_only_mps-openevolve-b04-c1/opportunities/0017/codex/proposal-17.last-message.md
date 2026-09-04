MECHANISM: Intermediate-horizon parameter-only EMA

HYPOTHESIS: An EMA interpolation rate of 0.03 will exceed 9,286 correct predictions by balancing the excess parameter lag at 0.01 against the reduced smoothing at 0.04.

INTENDED_EDIT: Increase the learned-parameter EMA rate from 0.02 to 0.03, shortening its effective horizon from roughly 50 to 33 optimizer steps.

EVIDENCE: Increasing the rate from 0.01 to 0.02 improved validation correct from 9,282 to 9,286, while increasing it further to 0.04 slightly regressed to 9,285; the untested midpoint directly probes the apparent optimum.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.03)
>>>>>>> REPLACE