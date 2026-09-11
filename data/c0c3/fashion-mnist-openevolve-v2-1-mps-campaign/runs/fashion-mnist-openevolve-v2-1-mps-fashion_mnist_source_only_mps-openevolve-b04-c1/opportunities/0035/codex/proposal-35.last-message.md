MECHANISM: Longer-horizon exponential weight averaging

HYPOTHESIS: Reducing the EMA interpolation rate from 0.02 to 0.015 will exceed 9,286 correct predictions by averaging roughly 67 rather than 50 terminal optimizer states, reducing late-training parameter noise without changing model capacity or runtime materially.

INTENDED_EDIT: Slow the existing final-half EMA update while preserving the architecture, objective, schedule, augmentation, and calibrated evaluation scale.

EVIDENCE: Classifier widening reduced validation correct to 9,254 and translation augmentation reduced it to 9,244, while temperature calibration has converged near 1.2112 without changing the 9,286 predictions; tuning the otherwise-unchanged EMA horizon is a focused next test for improving the primary accuracy criterion.

<<<<<<< SEARCH
                        averaged.lerp_(value.detach(), 0.02)
=======
                        averaged.lerp_(value.detach(), 0.015)
>>>>>>> REPLACE