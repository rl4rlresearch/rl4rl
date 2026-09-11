MECHANISM: Conservative peak-learning-rate increase

HYPOTHESIS: Increasing peak learning rate from 3.3e-3 to 3.5e-3 will exceed 9,249 correct predictions by improving convergence within the fixed 100,000-example budget.

INTENDED_EDIT: Raise only `PEAK_LR`, preserving the successful architecture, batch size, augmentation, schedule shape, and calibrated inference ensemble.

EVIDENCE: Prior higher peak learning rates improved correctness, while reducing batch size to 88, tail EMA, and lower dropout all regressed; this isolates a previously favorable optimization direction.

<<<<<<< SEARCH
PEAK_LR = 3.3e-3
=======
PEAK_LR = 3.5e-3
>>>>>>> REPLACE