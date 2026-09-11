MECHANISM: Finer-grained paired-view optimization

HYPOTHESIS: Halving the batch size to 32 with square-root learning-rate scaling will exceed 9,282 correct predictions by doubling optimizer updates while the paired-view forward pass still supplies 64 images per BatchNorm update.

INTENDED_EDIT: Reduce batch size from 64 to 32 and scale both the initial and scheduled AdamW learning rates from 2.1e-3 to 1.5e-3.

EVIDENCE: The prior 128-to-64 batch reduction increased validation correct from 9,212 to 9,215 with the same 216,250-parameter architecture; applying the same successful optimization change to the stronger paired-view, parameter-only-EMA design tests whether additional update frequency yields further gains.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.1e-3,
=======
        lr=1.5e-3,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.1e-3 * multiplier
=======
        group["lr"] = 1.5e-3 * multiplier
>>>>>>> REPLACE