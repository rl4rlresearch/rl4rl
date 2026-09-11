MECHANISM: Canonical-view weight boundary bisection retry

HYPOTHESIS: Setting each unshifted view’s weight to 1.06658935546875 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.06658935546875×, leaving shifted views unchanged.

EVIDENCE: Weight 1.0665283203125 preserved 9,290 correct, while 1.066650390625 lost one correct with slightly lower cross-entropy; their midpoint remains the most informative unresolved boundary probe because its only verification timed out.

<<<<<<< SEARCH
        view_weights = (1.0665283203125, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.06658935546875, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE