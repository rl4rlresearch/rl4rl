MECHANISM: Smaller-batch exposure-efficient optimization

HYPOTHESIS: Halving the batch size while linearly scaling the learning rate will exceed 9,050 correct predictions by doubling stochastic optimizer updates without changing the verified architecture or augmentation.

INTENDED_EDIT: Change batch size from 128 to 64 and scale both the initial and scheduled peak learning rates from 2.5e-4/2.5e-3 to 1.25e-4/1.25e-3.

EVIDENCE: The verified 244,386-parameter design reached 9,050 correct with 782 steps, improving on the earlier 392-step design; this isolates whether still more updates improve fixed-exposure learning.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-4,
=======
        lr=1.25e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 1.25e-3 * multiplier
>>>>>>> REPLACE