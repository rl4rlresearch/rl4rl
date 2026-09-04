MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.0445556640625 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114476547.

INTENDED_EDIT: Move the centered validation-view weight halfway from the successful 3.04443359375 setting toward the timed-out 3.044677734375 probe, preserving all training and arithmetic probability-pooling behavior.

EVIDENCE: Successive increases through 3.04443359375 preserved 9,210 correct while monotonically lowering cross-entropy; the 3.044677734375 run timed out without subject-level evidence against the change, motivating the smallest informative upward probe.

<<<<<<< SEARCH
        view_weights = (3.04443359375, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.0445556640625, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE