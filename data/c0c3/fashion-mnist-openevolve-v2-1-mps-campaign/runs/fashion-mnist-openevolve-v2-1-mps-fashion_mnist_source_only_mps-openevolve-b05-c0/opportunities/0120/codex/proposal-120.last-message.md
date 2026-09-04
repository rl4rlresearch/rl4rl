MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by emphasizing better-converged late states while retaining noise suppression.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.03 to 0.04.

EVIDENCE: Increasing EMA from 0.01 to 0.02 and then 0.03 added one correct prediction at each step and reduced cross-entropy; the prior 0.04 verification timed out without contradictory validation evidence.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.04)
>>>>>>> REPLACE