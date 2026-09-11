MECHANISM: Fine-grained centered-view boundary refinement

HYPOTHESIS: Raising the centered-view weight to 3.0446681976318359375 will preserve 9,210 correct predictions while reducing validation cross-entropy below 0.23114465255737304.

INTENDED_EDIT: Move halfway from the successful 3.044666290283203125 weight toward the timed-out 3.04467010498046875 probe, preserving all training and arithmetic probability-pooling behavior.

EVIDENCE: Successive increases through 3.044666290283203125 preserved 9,210 correct while monotonically lowering cross-entropy; the next larger probe timed out without subject-level evidence against it, so its midpoint is the most conservative informative step.

<<<<<<< SEARCH
        view_weights = (3.044666290283203125, 2.0, 2.0, 2.0, 2.0)
=======
        view_weights = (3.0446681976318359375, 2.0, 2.0, 2.0, 2.0)
>>>>>>> REPLACE