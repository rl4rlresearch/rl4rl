MECHANISM: Intermediate-batch higher-update optimization

HYPOTHESIS: Reducing batch size from 48 to 40 will exceed 9,280 correct predictions by increasing optimizer steps from 2,084 to approximately 2,500 while remaining faster than the batch-32 configuration that timed out.

INTENDED_EDIT: Change only the training batch size from 48 to 40, preserving the proven architecture, augmentation, optimizer, schedule, and validation ensemble.

EVIDENCE: Reducing batch size from 64 to 48 increased optimizer steps by 33% and improved validation_correct from 9,238 to 9,280; batch size 32 timed out, so 40 is the next conservative point between the best completed run and the runtime boundary.

<<<<<<< SEARCH
BATCH_SIZE = 48
=======
BATCH_SIZE = 40
>>>>>>> REPLACE