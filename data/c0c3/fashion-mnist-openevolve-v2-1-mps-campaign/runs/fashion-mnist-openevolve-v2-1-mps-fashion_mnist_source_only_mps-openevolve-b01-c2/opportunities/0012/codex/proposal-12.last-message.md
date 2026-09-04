MECHANISM: Time-budget-aware intermediate batch scaling

HYPOTHESIS: Reducing the unique-image batch size to 48 will exceed 9,312 correct predictions by increasing optimizer updates beyond batch size 64 while remaining fast enough to avoid batch size 32’s verification timeout.

INTENDED_EDIT: Keep the proven architecture, paired-flip objective, and schedule unchanged while reducing `BATCH_SIZE` from 128 to 48.

EVIDENCE: Batch size 64 improved the same implementation from 9,310 to 9,312 correct and reduced cross-entropy from 0.20765 to 0.20389; batch size 32 timed out, motivating an intermediate 48-image batch.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 48
>>>>>>> REPLACE