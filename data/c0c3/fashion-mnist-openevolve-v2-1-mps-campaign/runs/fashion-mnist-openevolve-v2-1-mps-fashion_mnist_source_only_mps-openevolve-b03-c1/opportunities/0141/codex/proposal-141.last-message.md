MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.044921875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114547729.

INTENDED_EDIT: Move the centered validation-view weight halfway from the successful 3.04296875 setting toward the inconclusive 3.046875 setting, preserving all training and probability-pooling behavior.

EVIDENCE: Increasing the weight from 3.0390625 to 3.04296875 preserved 9,210 correct and lowered cross-entropy from 0.23114737778 to 0.23114547729; the lower-cross-entropy result at 3.125 supports continuing upward cautiously before its one-prediction loss.

<<<<<<< SEARCH
        view_weights = (3.04296875, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.044921875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE