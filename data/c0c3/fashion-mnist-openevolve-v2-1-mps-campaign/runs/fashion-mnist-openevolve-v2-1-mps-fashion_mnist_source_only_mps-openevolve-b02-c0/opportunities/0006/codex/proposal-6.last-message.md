MECHANISM: Conservative optimizer-step densification

HYPOTHESIS: Reducing batch size from 48 to 44 will exceed 9,146 correct predictions by providing about 9% more optimizer updates while remaining within the runtime limit that batch size 32 exceeded.

INTENDED_EDIT: Change only the training batch size from 48 to 44, preserving the best-performing architecture, augmentation, optimizer, and schedule.

EVIDENCE: Reducing batch size from 64 to 48 increased optimizer steps from 1,564 to 2,084 and validation_correct from 9,141 to 9,146, while batch size 32 timed out; 44 is the nearest conservative test of the demonstrated update-density trend.

<<<<<<< SEARCH
BATCH_SIZE = 48
=======
BATCH_SIZE = 44
>>>>>>> REPLACE