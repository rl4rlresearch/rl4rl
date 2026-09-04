MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.0439453125 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114547729.

INTENDED_EDIT: Move the centered validation-view weight halfway from the successful 3.04296875 setting toward the inconclusive 3.044921875 setting, preserving all training and arithmetic probability-pooling behavior.

EVIDENCE: Successive increases from 3.03125 through 3.04296875 preserved 9,210 correct while monotonically lowering cross-entropy; 3.125 lost one prediction, so a minimal upward probe is justified.

<<<<<<< SEARCH
        view_weights = (3.04296875, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.0439453125, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE