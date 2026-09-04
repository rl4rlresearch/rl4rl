MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.04443359375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114500771.

INTENDED_EDIT: Move the centered validation-view weight halfway from the successful 3.0439453125 setting toward the inconclusive 3.044921875 setting, preserving all training and probability-pooling behavior.

EVIDENCE: Successive increases through 3.0439453125 preserved 9,210 correct while monotonically lowering cross-entropy; the 3.044921875 verification timed out, making its midpoint the smallest informative upward probe.

<<<<<<< SEARCH
        view_weights = (3.0439453125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.04443359375, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE