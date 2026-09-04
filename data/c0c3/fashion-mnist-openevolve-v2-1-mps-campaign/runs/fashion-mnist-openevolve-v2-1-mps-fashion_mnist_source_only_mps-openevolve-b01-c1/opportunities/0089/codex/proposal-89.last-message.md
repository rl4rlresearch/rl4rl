MECHANISM: Smaller-batch, learning-rate-scaled optimization

HYPOTHESIS: Using batch size 96 with a proportionally moderated learning rate will exceed 9,247 correct predictions by providing 33% more optimizer updates and beneficial gradient noise without increasing total example exposure.

INTENDED_EDIT: Reduce batch size from 128 to 96 and scale the initial and peak learning rates by approximately √(96/128).

EVIDENCE: Recent crop, ensemble, consistency, EMA, and capacity changes all regressed, motivating a shift from invariance and architecture tuning to the still-unexplored optimization granularity.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 96
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
        weight_decay=2.0e-4,
=======
        lr=2.6e-4,
        weight_decay=2.0e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 2.6e-3 * multiplier
>>>>>>> REPLACE