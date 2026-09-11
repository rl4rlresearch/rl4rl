MECHANISM: Smaller-batch optimization under fixed exposure

HYPOTHESIS: Reducing the batch size from 128 to 64 will exceed 9,159 correct predictions by providing about 1,563 optimizer updates instead of 782 while preserving the successful architecture, augmentation, schedule, and cardinal-view ensemble.

INTENDED_EDIT: Halve the training batch size without changing any other training or inference behavior.

EVIDENCE: The earlier change that doubled optimizer updates from 392 to 782 was part of an improvement from 8,928 to 9,134 correct; isolating another update-frequency increase is the most informative next test after additional diagonal inference views failed.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE