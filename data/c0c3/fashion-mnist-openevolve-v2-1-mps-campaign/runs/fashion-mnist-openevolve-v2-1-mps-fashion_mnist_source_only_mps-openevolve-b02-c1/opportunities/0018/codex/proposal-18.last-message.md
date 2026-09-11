MECHANISM: Runtime-bracketed intermediate-batch optimization

HYPOTHESIS: Batch size 44 will exceed 9,280 correct predictions by providing approximately 9% more optimizer steps than batch size 48 while remaining within the runtime boundary that batch size 40 exceeded.

INTENDED_EDIT: Reduce only the training batch size from 48 to 44, preserving the architecture, augmentation, optimizer, schedule, and validation ensemble.

EVIDENCE: Reducing batch size from 64 to 48 increased validation_correct from 9,238 to 9,280, while batch size 40 timed out; 44 is the nearest conservative midpoint between the best completed configuration and the observed runtime failure.

<<<<<<< SEARCH
BATCH_SIZE = 48
=======
BATCH_SIZE = 44
>>>>>>> REPLACE