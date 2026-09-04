MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Raising the EMA coefficient to 0.03 will increase validation_correct above 9,259 by tracking the better-converged final training states more closely while retaining noise suppression.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.02 to 0.03.

EVIDENCE: Increasing EMA from 0.01 to 0.02 improved validation_correct from 9,258 to 9,259 and cross-entropy from 0.2135168 to 0.2126968, motivating the next controlled step in the same direction.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.02)
=======
                    average.lerp_(parameter.detach(), 0.03)
>>>>>>> REPLACE