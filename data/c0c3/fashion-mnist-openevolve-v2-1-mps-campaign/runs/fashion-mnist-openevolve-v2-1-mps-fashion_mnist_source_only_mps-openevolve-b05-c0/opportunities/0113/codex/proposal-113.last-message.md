MECHANISM: Faster late-training exponential weight averaging

HYPOTHESIS: Increasing the EMA update coefficient from 0.01 to 0.02 will raise validation_correct above 9,258 by emphasizing better-converged late checkpoints while retaining enough averaging to suppress optimizer noise.

INTENDED_EDIT: Halve the effective EMA lag without changing architecture, training exposure, evaluation cost, or parameter count.

EVIDENCE: Evaluation-logit calibration improved only cross-entropy while preserving all 9,258 predictions, so the next change must alter learned decision boundaries; the current halfway-started EMA still averages substantially older states in a short 3,126-step run.

<<<<<<< SEARCH
                    average.lerp_(parameter.detach(), 0.01)
=======
                    average.lerp_(parameter.detach(), 0.02)
>>>>>>> REPLACE