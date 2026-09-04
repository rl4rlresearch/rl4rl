MECHANISM: Higher-update stochastic optimization

HYPOTHESIS: Reducing the batch size from 64 to 32 will exceed 9,238 correct predictions by doubling optimizer updates again while retaining adequate batch-normalization statistics.

INTENDED_EDIT: Set the training batch size to 32 while preserving the architecture, augmentation, optimizer, and exposure-relative learning-rate schedule.

EVIDENCE: The controlled reduction from batch size 128 to 64 doubled optimizer steps from 782 to 1,564 and improved validation_correct from 9,232 to 9,238 while lowering cross-entropy from 0.20637 to 0.20281, directly motivating another batch-size reduction.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE