MECHANISM: Fixed-budget learning-rate intensification

HYPOTHESIS: Raising the peak learning rate by approximately 6% will exceed 9,345 correct predictions by extracting more optimization progress from the fixed 1,042 updates without increasing parameters, computation, or evaluation cost.

INTENDED_EDIT: Increase `PEAK_LR` from 3.3e-3 to 3.5e-3 while preserving the architecture, batch size, augmentation, EMA, and cosine schedule.

EVIDENCE: The raw-mean gated architecture reached the best 9,345-correct result, while added representation paths regressed or timed out; the smaller-batch experiment specifically identified update density as a plausible limitation but increased runtime, motivating a modest zero-cost increase in update magnitude.

<<<<<<< SEARCH
PEAK_LR = 3.3e-3
=======
PEAK_LR = 3.5e-3
>>>>>>> REPLACE