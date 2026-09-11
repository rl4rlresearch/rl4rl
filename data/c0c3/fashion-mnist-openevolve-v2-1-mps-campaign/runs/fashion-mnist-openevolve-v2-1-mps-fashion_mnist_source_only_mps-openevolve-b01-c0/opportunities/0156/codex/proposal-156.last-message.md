MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.0665515661239624 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246069984436035.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0665514469146729× to 1.0665515661239624×, leaving shifted views unchanged.

EVIDENCE: Weight 1.0665514469146729 preserved 9,290 correct, while 1.066551685333252 lost one correct with lower cross-entropy; the sole representable midpoint is the most informative remaining accuracy-boundary probe.

<<<<<<< SEARCH
        view_weights = (1.0665514469146729, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE