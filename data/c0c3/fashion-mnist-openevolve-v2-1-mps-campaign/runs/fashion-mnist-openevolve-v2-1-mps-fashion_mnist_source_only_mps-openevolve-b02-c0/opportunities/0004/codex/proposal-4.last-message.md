MECHANISM: Further small-batch optimizer-step densification

HYPOTHESIS: Reducing batch size from 64 to 32 while preserving the proven architecture and recipe will exceed 9,141 correct predictions by providing roughly twice as many parameter updates within the fixed 100,000-example exposure.

INTENDED_EDIT: Change only the training batch size from 64 to 32.

EVIDENCE: Reducing batch size from 128 to 64 increased optimizer steps from 782 to 1,564 and improved validation_correct from 9,128 to 9,141, while the architecture-changing residual alternative fell to 8,985; this supports another controlled update-density test.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE