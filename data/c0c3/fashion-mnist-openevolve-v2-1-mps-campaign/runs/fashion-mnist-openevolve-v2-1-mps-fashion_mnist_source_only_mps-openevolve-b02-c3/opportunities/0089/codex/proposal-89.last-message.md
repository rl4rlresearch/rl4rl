MECHANISM: Continued vertical-favoring translation ensemble

HYPOTHESIS: Increasing each vertical-crop weight to 1.375 and decreasing each horizontal-crop weight to 0.625 will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.

INTENDED_EDIT: Shift another 0.0625 of weight from each horizontal crop to each vertical crop while preserving the center weight and total ensemble weight.

EVIDENCE: Correctness improved monotonically from 9,344 at symmetric 1.0/1.0 weights to 9,346 at 1.25/0.75 and 9,348 at 1.3125/0.6875, motivating one more equal-sized step in the successful direction.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
=======
        crop_weights = (3.0, 1.375, 1.375, 0.625, 0.625)
>>>>>>> REPLACE