MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.04467010498046875 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465370.

INTENDED_EDIT: Move halfway from the successful 3.0446624755859375 weight toward the timed-out 3.044677734375 probe, preserving training and arithmetic probability pooling.

EVIDENCE: Successive increases through 3.0446624755859375 preserved 9,210 correct while monotonically lowering cross-entropy; the larger 3.044677734375 probe timed out without evidence against the change, so its midpoint is the most conservative informative step.

<<<<<<< SEARCH
        view_weights = (3.0446624755859375, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.04467010498046875, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE