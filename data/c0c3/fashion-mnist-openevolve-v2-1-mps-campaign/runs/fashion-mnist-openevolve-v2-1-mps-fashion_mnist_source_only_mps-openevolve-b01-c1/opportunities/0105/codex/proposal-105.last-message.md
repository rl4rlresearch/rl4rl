MECHANISM: Smaller-batch, linearly scaled AdamW optimization

HYPOTHESIS: Reducing the batch size to 96 while scaling the learning rate by 0.75 will exceed 9,257 correct predictions by providing 1,042 less-correlated optimizer updates without increasing cumulative learning-rate exposure.

INTENDED_EDIT: Change batch size from 128 to 96 and scale both the initial and scheduled peak learning rates from 3.0e-4/3.0e-3 to 2.25e-4/2.25e-3.

EVIDENCE: The 233,434-parameter architecture remains best at 9,257 correct, while further widening, added pooling, dropout, global readout, and alternative aggregation all regressed; this preserves the proven model and tests the largely unexplored optimization granularity axis.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 96
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
        weight_decay=2.0e-4,
=======
        lr=2.25e-4,
        weight_decay=2.0e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 2.25e-3 * multiplier
>>>>>>> REPLACE