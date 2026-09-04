MECHANISM: Finer-grained fixed-exposure optimization

HYPOTHESIS: Reducing the batch size from 64 to 50 will exceed 9,290 correct predictions by increasing optimizer updates from 1,564 to exactly 2,000 while preserving the validated model, augmentation, loss, schedule, and parameter count.

INTENDED_EDIT: Use 50-example minibatches without changing any other training or evaluation behavior.

EVIDENCE: The 9,290-correct design received only 1,564 optimizer updates, while recent architectural, regularization, and ensemble changes regressed or timed out; update granularity remains untested and can increase learning opportunities without adding parameters or expensive computation.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 2.0
=======
BATCH_SIZE = 50
GRAD_CLIP_NORM = 2.0
>>>>>>> REPLACE