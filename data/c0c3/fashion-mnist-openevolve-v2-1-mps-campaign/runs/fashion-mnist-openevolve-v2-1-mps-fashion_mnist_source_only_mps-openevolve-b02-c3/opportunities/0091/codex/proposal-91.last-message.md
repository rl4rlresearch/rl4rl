MECHANISM: Bracketed vertical-shift weight refinement

HYPOTHESIS: Midpoint weights of 1.34375 vertical and 0.65625 horizontal will exceed 9,348 correct predictions or retain 9,348 while lowering cross-entropy below 0.18770758.

INTENDED_EDIT: Restore the best TTA direction and test the midpoint between the best 1.3125/0.6875 weights and the regressive 1.375/0.625 weights, preserving center and total weight.

EVIDENCE: Vertical weighting improved from 9,346 correct at 1.25 to 9,348 at 1.3125, then declined to 9,347 at 1.375, directly bracketing a likely optimum.

<<<<<<< SEARCH
        crop_weights = (3.0, 1.0, 1.0, 1.0, 1.0)
=======
        crop_weights = (3.0, 1.34375, 1.34375, 0.65625, 0.65625)
>>>>>>> REPLACE