MECHANISM: Exact-divisor smaller minibatches

HYPOTHESIS: Using batch size 50 will exceed 9,290 correct predictions by increasing optimizer updates from 1,564 to exactly 2,000 while eliminating undersized end-of-epoch batches.

INTENDED_EDIT: Change the batch size from 64 to 50, preserving the successful model, augmentation, optimizer, schedule, and ensemble.

EVIDENCE: The 247,836-parameter design achieved 9,290 correct in 1,564 steps, while multiple architecture and regularization changes regressed; increasing update resolution without altering those validated components is the most informative remaining optimization change.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 50
>>>>>>> REPLACE