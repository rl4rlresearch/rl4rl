MECHANISM: Conservative centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.046875 will preserve 9,210 correct predictions while further lowering validation cross-entropy.

INTENDED_EDIT: Move the centered validation-view weight one small step toward the lower-cross-entropy 3.125 setting, preserving arithmetic probability pooling and all training behavior.

EVIDENCE: Weight 3.03125 retained 9,210 correct and improved cross-entropy over 3.0, while 3.125 lowered cross-entropy further but lost one prediction; a smaller interpolation tests the supported calibration direction with limited boundary risk.

<<<<<<< SEARCH
        view_weights = (3.03125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.046875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE