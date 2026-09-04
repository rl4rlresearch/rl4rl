MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.044647216796875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114468079.

INTENDED_EDIT: Move halfway from the successful 3.04461669921875 weight toward the timed-out 3.044677734375 probe, preserving all training and arithmetic probability-pooling behavior.

EVIDENCE: Successive increases through 3.04461669921875 preserved 9,210 correct while monotonically lowering cross-entropy; the larger 3.044677734375 probe timed out without subject-level evidence against it, motivating the smallest informative upward step.

<<<<<<< SEARCH
        view_weights = (3.04461669921875, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.044647216796875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE