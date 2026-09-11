MECHANISM: Slightly higher-update small-batch optimization

HYPOTHESIS: Reducing batch size from 32 to 31 will exceed 9,254 correct predictions by providing about 100 additional optimizer updates while remaining close to the proven runtime and EMA exposure horizon.

INTENDED_EDIT: Change only the training batch size from 32 to 31.

EVIDENCE: Reducing batch size from 128 to 64 to 32 monotonically improved correct predictions from 9,210 to 9,235 to 9,250; this minimal reduction tests the same mechanism with substantially less runtime risk than the timed-out batch-24 and batch-28 designs.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 31
>>>>>>> REPLACE