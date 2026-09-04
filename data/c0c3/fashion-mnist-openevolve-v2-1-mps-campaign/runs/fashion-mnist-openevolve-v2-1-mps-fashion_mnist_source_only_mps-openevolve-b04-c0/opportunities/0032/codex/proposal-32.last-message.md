MECHANISM: Upper-side decoupled parameter-EMA refinement

HYPOTHESIS: A 4.25% parameter EMA update with the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly improving weight responsiveness without disturbing the BatchNorm smoothing responsible for the current accuracy gain.

INTENDED_EDIT: Increase only the parameter EMA update rate from 4% to 4.25%, preserving the 3% floating-buffer update and all other behavior.

EVIDENCE: Reducing the parameter update to 3.75% with a 3% buffer update regressed from 9,359 to 9,358 correct; the symmetric upper-side refinement is the closest informative untested direction around the best 4% parameter rate.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
=======
                    average.lerp_(parameter.detach(), 0.0425)
>>>>>>> REPLACE