MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by tracking later, better-converged training states while retaining substantial noise suppression.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.03 to 0.04.

EVIDENCE: Successive EMA increases from 0.01 to 0.02 and then 0.03 each added one correct prediction, reaching 9,260, while also reducing cross-entropy from 0.2135168 to 0.2125679; this motivates one more controlled step in the same direction.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.04)
>>>>>>> REPLACE