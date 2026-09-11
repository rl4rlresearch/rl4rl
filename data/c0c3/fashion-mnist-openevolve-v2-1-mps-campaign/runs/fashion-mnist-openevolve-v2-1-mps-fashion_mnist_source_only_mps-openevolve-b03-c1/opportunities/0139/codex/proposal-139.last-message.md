MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.04296875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114737778.

INTENDED_EDIT: Move the centered validation-view weight halfway from the successful 3.0390625 setting toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.

EVIDENCE: Increasing the weight from 3.03125 to 3.0390625 preserved 9,210 correct and lowered cross-entropy from 0.23115117645 to 0.23114737778; 3.125 lowered cross-entropy further but lost one prediction, motivating a smaller upward boundary probe.

<<<<<<< SEARCH
        view_weights = (3.0390625, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.04296875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE