MECHANISM: Intermediate small-batch optimizer-step densification

HYPOTHESIS: Batch size 48 will exceed 9,141 correct predictions by increasing optimizer updates beyond batch size 64 while avoiding the verification timeout observed at batch size 32.

INTENDED_EDIT: Change only the training batch size from 64 to 48, preserving the proven architecture, augmentation, optimizer, and schedule.

EVIDENCE: Reducing batch size from 128 to 64 increased optimizer steps from 782 to 1,564 and improved validation_correct from 9,128 to 9,141, while batch size 32 did not finish; batch size 48 tests the supported update-density trend within a safer runtime envelope.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 48
>>>>>>> REPLACE