MECHANISM: Intermediate-frequency optimization with exposure-matched regularization and EMA

HYPOTHESIS: Batch size 48 will exceed 9,128 correct predictions by adding one-third more optimizer updates than batch 64 while avoiding the verification timeout encountered at batch 32.

INTENDED_EDIT: Use batch size 48 and scale AdamW weight decay and EMA interpolation proportionally to preserve their behavior per training example.

EVIDENCE: Reducing batch size from 128 to 64 improved correct predictions from 9,125 to 9,128 and cross-entropy from 0.24994 to 0.24579, while batch 32 timed out; batch 48 tests an intermediate update/runtime tradeoff.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 48
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight_decay=3e-4,
=======
        weight_decay=1.125e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
                average.lerp_(tensor, 0.01)
=======
                average.lerp_(tensor, 0.00375)
>>>>>>> REPLACE