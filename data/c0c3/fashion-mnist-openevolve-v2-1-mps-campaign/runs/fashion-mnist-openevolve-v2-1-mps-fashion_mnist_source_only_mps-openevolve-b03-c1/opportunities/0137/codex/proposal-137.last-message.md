MECHANISM: Conservative centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.0390625 will preserve 9,210 correct predictions while lowering validation cross-entropy below 0.23115117645.

INTENDED_EDIT: Move halfway from the successful 3.03125 weight toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.

EVIDENCE: Weight 3.03125 retained 9,210 correct while improving cross-entropy over 3.0; weight 3.125 lowered cross-entropy further but lost one prediction, supporting a small upward calibration step.

<<<<<<< SEARCH
        view_weights = (3.03125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.0390625, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE