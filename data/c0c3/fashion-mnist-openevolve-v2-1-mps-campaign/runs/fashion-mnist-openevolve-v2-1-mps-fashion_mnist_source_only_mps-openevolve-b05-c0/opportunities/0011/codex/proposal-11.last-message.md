MECHANISM: Higher update density from smaller minibatches

HYPOTHESIS: Reducing batch size from 64 to 32 will exceed 9,235 correct predictions by increasing optimizer steps from 1,564 to approximately 3,126 within the fixed exposure budget.

INTENDED_EDIT: Change only the training batch size, preserving the successful BatchNorm architecture, label-smoothed loss, cosine schedule, and tail EMA.

EVIDENCE: Reducing batch size from 128 to 64 increased validation correct from 9,210 to 9,235 while holding the rest of the implementation fixed, directly motivating the next isolated reduction.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE