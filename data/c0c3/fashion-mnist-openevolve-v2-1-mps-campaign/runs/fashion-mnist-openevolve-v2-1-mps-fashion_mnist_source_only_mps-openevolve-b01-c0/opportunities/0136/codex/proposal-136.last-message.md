MECHANISM: Canonical-view weight boundary bisection retry

HYPOTHESIS: Setting each unshifted view’s weight to 1.09375 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.09375× while leaving shifted views unchanged.

EVIDENCE: Weight 1.0625 preserved 9,290 correct, whereas 1.125 lost two correct but improved cross-entropy; the midpoint probe timed out, so retrying it is the most informative unresolved boundary test.

<<<<<<< SEARCH
        view_weights = (1.0625, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.09375, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE