MECHANISM: Update-budget-compensated learning-rate scaling

HYPOTHESIS: Raising the peak learning rate from 3.0e-3 to 3.3e-3 will exceed 9,239 correct predictions by approximating the greater cumulative optimization progress sought with batch size 88, without its runtime increase.

INTENDED_EDIT: Increase the learning-rate schedule uniformly by 10% while retaining batch size 96 and all verified architecture, augmentation, and inference behavior.

EVIDENCE: Reducing batch size from 128 to 96 improved correctness from 9,204 to 9,239 through more optimizer updates, while batch size 88 timed out; scaling 3.0e-3 by the attempted update-count ratio of approximately 1,137/1,042 gives about 3.27e-3, motivating the conservative 3.3e-3 value.

<<<<<<< SEARCH
PEAK_LR = 3.0e-3
=======
PEAK_LR = 3.3e-3
>>>>>>> REPLACE