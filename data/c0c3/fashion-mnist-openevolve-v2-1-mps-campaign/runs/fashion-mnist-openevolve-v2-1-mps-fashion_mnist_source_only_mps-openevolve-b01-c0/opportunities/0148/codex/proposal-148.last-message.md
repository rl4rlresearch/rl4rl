MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.066558837890625 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024607120513916.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0665283203125× to 1.066558837890625×, leaving shifted views unchanged.

EVIDENCE: Weight 1.0665283203125 preserved 9,290 correct, while 1.06658935546875 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining probe of the accuracy-preserving boundary.

<<<<<<< SEARCH
        view_weights = (1.0665283203125, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.066558837890625, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE