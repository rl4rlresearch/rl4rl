MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Raising the EMA coefficient to 0.04 will increase validation_correct above 9,260 by tracking later training states more closely while retaining noise suppression.

INTENDED_EDIT: Increase only the EMA interpolation coefficient from 0.03 to 0.04.

EVIDENCE: EMA increases from 0.01 to 0.02 and 0.03 each added one correct prediction and reduced cross-entropy; two 0.04 attempts timed out without contradictory accuracy evidence.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.03)
=======
                    average.lerp_(parameter.detach(), 0.04)
>>>>>>> REPLACE