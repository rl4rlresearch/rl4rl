MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.068359375 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246076278686523.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.06640625× to 1.068359375×, leaving shifted views unchanged.

EVIDENCE: Weight 1.06640625 preserved 9,290 correct with 0.20246076278686523 cross-entropy, while 1.0703125 lost one correct but lowered cross-entropy; their midpoint is the most informative probe of the accuracy-preserving boundary.

<<<<<<< SEARCH
        view_weights = (1.06640625, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.068359375, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE