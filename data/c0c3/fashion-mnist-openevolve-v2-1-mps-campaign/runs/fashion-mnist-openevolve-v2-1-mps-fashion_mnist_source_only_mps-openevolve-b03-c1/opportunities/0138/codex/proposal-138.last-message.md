MECHANISM: Conservative centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.046875 will preserve 9,210 correct predictions while lowering validation cross-entropy below 0.23114737778.

INTENDED_EDIT: Increase only the centered validation-view weight, preserving training and arithmetic probability pooling.

EVIDENCE: Increasing the weight from 3.03125 to 3.0390625 preserved 9,210 correct and reduced cross-entropy from 0.23115117645 to 0.23114737778; 3.125 reduced cross-entropy further but lost one prediction, motivating another small step toward that boundary.

<<<<<<< SEARCH
        view_weights = (3.0390625, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.046875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE