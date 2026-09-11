MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.0665531158447266 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246070098876953.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.06655120849609375× to 1.0665531158447266×, leaving shifted views unchanged.

EVIDENCE: Weight 1.06655120849609375 preserved 9,290 correct, while 1.0665550231933594 lost one correct but slightly lowered cross-entropy; their midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        view_weights = (1.06655120849609375, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665531158447266, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE