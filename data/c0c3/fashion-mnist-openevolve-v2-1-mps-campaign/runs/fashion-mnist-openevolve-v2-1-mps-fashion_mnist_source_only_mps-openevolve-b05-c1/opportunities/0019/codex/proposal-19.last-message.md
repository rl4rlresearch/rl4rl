MECHANISM: Interpolated peak-rate calibration

HYPOTHESIS: Setting the peak learning rate to 3.15e-3 will exceed 9,240 correct predictions by retaining most of the optimization benefit of 3.3e-3 while reducing its apparent late-stage overshoot.

INTENDED_EDIT: Reduce only the peak learning rate from 3.3e-3 to the midpoint between the two verified rates, preserving the successful batch size, warmup, architecture, augmentation, and inference ensemble.

EVIDENCE: At batch size 96, increasing the peak rate from 3.0e-3 to 3.3e-3 gained one correct prediction but worsened cross-entropy from 0.21392 to 0.21460; the midpoint directly tests whether the accuracy gain can be retained with better convergence.

<<<<<<< SEARCH
PEAK_LR = 3.3e-3
=======
PEAK_LR = 3.15e-3
>>>>>>> REPLACE