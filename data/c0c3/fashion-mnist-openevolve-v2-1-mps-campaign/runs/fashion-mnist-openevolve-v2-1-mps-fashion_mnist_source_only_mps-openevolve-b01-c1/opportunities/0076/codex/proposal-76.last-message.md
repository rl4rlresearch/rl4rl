MECHANISM: Smaller-batch, linearly scaled optimization

HYPOTHESIS: Reducing the batch size from 128 to 64 while halving the learning-rate scale will exceed 9,247 correct predictions by doubling optimizer updates and batch-shared crop draws without increasing fixed-example exposure or changing the established augmentation geometry.

INTENDED_EDIT: Use batch size 64, a 1.5e-4 initial learning rate, and a 1.5e-3 scheduled peak, preserving the learning-rate trajectory per processed example.

EVIDENCE: Per-example translation sampling regressed to 9,219, suggesting batch-correlated views are beneficial; smaller batches preserve that correlation while increasing crop-offset coverage and optimization granularity, an axis not tested by the recent architecture, regularization, or loss-schedule changes.

<<<<<<< SEARCH
BATCH_SIZE = 128
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=3.0e-4,
        weight_decay=2.0e-4,
=======
        lr=1.5e-4,
        weight_decay=2.0e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 3.0e-3 * multiplier
=======
        group["lr"] = 1.5e-3 * multiplier
>>>>>>> REPLACE