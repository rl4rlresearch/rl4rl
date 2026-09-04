MECHANISM: Increased fixed-exposure update frequency

HYPOTHESIS: Reducing batch size from 96 to 80 will exceed 9,249 correct predictions by providing 20% more optimizer updates and beneficial gradient noise within the fixed 100,000-example budget.

INTENDED_EDIT: Change only the training batch size, preserving the architecture, learning-rate schedule, augmentation, and calibrated probability-space ensemble.

EVIDENCE: A prior batch-size reduction improved correctness from 9,204 to 9,239, whereas subsequent architectural changes reduced correctness; this directly supports continuing the successful optimization direction conservatively.

<<<<<<< SEARCH
BATCH_SIZE = 96
=======
BATCH_SIZE = 80
>>>>>>> REPLACE