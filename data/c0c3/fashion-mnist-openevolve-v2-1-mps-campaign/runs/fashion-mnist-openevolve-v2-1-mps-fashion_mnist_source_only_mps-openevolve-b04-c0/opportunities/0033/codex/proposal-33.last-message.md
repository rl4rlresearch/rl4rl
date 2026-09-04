MECHANISM: Conservative upper-side parameter-EMA refinement

HYPOTHESIS: A 4.125% parameter EMA update with the best-tested 3% buffer update will exceed 9,359 correct predictions—or tie with lower cross-entropy—by modestly increasing weight responsiveness without disturbing BatchNorm smoothing.

INTENDED_EDIT: Increase only the parameter EMA update rate from 4% to 4.125%.

EVIDENCE: Reducing the parameter update to 3.75% regressed to 9,358 correct, while 4% achieved 9,359; the 4.25% test timed out, making 4.125% the closest evaluable upper-side refinement.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.04)
=======
                    average.lerp_(parameter.detach(), 0.04125)
>>>>>>> REPLACE