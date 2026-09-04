MECHANISM: Moderate fixed-exposure update-frequency increase

HYPOTHESIS: Reducing batch size from 96 to 88 will exceed 9,249 correct predictions by providing about 9% more optimizer updates while avoiding the greater runtime risk observed at batch size 80.

INTENDED_EDIT: Change only the training batch size, preserving the model, learning-rate schedule, augmentation, and calibrated inference ensemble.

EVIDENCE: A prior batch-size reduction improved correctness from 9,204 to 9,239, while the batch-size-80 experiment timed out; 88 is a conservative intermediate test of the same favorable optimization direction.

<<<<<<< SEARCH
BATCH_SIZE = 96
=======
BATCH_SIZE = 88
>>>>>>> REPLACE