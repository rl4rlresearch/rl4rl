MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA coefficient from 0.01 to 0.02 will raise validation_correct above 9,258 by emphasizing better-converged late checkpoints while retaining substantial noise suppression.

INTENDED_EDIT: Double the EMA update coefficient without changing architecture, training exposure, evaluation cost, or parameter count.

EVIDENCE: Logit calibration improved cross-entropy but preserved exactly 9,258 predictions, so further gains require changed decision boundaries; the current EMA begins halfway through the short 3,126-step run and updates only every four steps, making 0.01 strongly favor older states. The prior 0.02 verification timed out and supplied no contradictory result.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.02)
>>>>>>> REPLACE