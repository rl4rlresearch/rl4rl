MECHANISM: Intermediate smaller-batch optimization

HYPOTHESIS: Reducing batch size from 128 to 96 will exceed 9,204 correct predictions by increasing optimizer updates from 782 to about 1,042 while avoiding the runtime cost that caused batch size 64 to time out.

INTENDED_EDIT: Change only the training batch size from 128 to 96, preserving the verified architecture, matched center/cardinal augmentation, schedule, and inference ensemble.

EVIDENCE: The 9,204-correct design is the strongest verified implementation; batch size 64 was motivated by additional optimizer updates but timed out, so 96 is the most informative intermediate runtime–update tradeoff.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 96
>>>>>>> REPLACE