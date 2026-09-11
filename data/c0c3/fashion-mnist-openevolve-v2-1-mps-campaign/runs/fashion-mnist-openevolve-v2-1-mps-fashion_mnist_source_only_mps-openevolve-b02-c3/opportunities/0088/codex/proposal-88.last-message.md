MECHANISM: Fine-grained vertical-favoring translation ensemble

HYPOTHESIS: Increasing the vertical-shift preference slightly beyond the best 1.25/0.75 weighting will exceed 9,346 correct predictions or retain 9,346 while lowering cross-entropy below 0.18771290.

INTENDED_EDIT: Restore the proven center weight of 3.0 and shift another 0.0625 of weight from each horizontal crop to each vertical crop, preserving total ensemble weight.

EVIDENCE: Symmetric 1.0/1.0 shift weights achieved 9,344 correct, horizontal-favoring 0.75/1.25 fell to 9,339, and vertical-favoring 1.25/0.75 improved to 9,346; a small continuation tests whether the optimum lies just beyond the best observed asymmetry.

<<<<<<< SEARCH
        crop_weights = (2.75, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 1.3125, 1.3125, 0.6875, 0.6875)
>>>>>>> REPLACE