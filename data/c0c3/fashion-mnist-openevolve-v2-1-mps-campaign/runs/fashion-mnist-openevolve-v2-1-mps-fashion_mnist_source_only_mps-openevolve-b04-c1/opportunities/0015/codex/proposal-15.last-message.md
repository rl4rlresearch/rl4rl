MECHANISM: Intermediate small-batch paired-view optimization

HYPOTHESIS: Batch size 48 with square-root-scaled learning rate will exceed 9,286 correct predictions by providing 33% more optimizer updates than batch 64 while avoiding the timeout observed at batch 32.

INTENDED_EDIT: Reduce batch size from 64 to 48 and scale the AdamW base and scheduled learning rates from 2.1e-3 to 1.8e-3.

EVIDENCE: Reducing batch size from 128 to 64 improved correct predictions from 9,212 to 9,215, while batch 32 timed out; batch 48 tests an intermediate update-frequency increase on the stronger paired-view, short-horizon EMA design.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 48
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.1e-3,
=======
        lr=1.8e-3,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.1e-3 * multiplier
=======
        group["lr"] = 1.8e-3 * multiplier
>>>>>>> REPLACE