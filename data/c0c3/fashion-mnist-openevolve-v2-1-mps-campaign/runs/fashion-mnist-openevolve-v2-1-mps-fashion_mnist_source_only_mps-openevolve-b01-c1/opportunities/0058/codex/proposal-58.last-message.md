MECHANISM: Smaller-minibatch update densification

HYPOTHESIS: Reducing the batch size from 128 to 96 will exceed 9,247 correct predictions by providing 1,042 optimizer updates instead of 782 under the same fixed exposure and cosine curricula, improving convergence without changing the proven model or aggregation objective.

INTENDED_EDIT: Set the training batch size to 96 while preserving architecture, losses, augmentation, optimizer, and progress-normalized schedules.

EVIDENCE: The 25%-to-50% cosine pair curriculum is best at 9,247 correct, while further loss-schedule and crop-sampling refinements tied or regressed; every reported verification used batch size 128 and 782 steps, leaving optimizer-update density as an informative untested axis.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 96
>>>>>>> REPLACE