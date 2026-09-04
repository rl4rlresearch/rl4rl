MECHANISM: Moderately smaller-batch optimization

HYPOTHESIS: Reducing batch size from 96 to 88 will exceed 9,239 correct predictions by increasing optimizer steps from 1,042 to about 1,137 while remaining substantially closer to the verified runtime than the timed-out batch-size-64 design.

INTENDED_EDIT: Change only the training batch size from 96 to 88, preserving the architecture, optimizer, schedule, matched center/cardinal augmentation, and inference ensemble.

EVIDENCE: Reducing batch size from 128 to 96 improved validation correctness from 9,204 to 9,239, whereas 64 timed out; 88 is a conservative continuation of the demonstrated update-frequency benefit.

<<<<<<< SEARCH
BATCH_SIZE = 96
=======
BATCH_SIZE = 88
>>>>>>> REPLACE