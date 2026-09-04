MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.09375 will retain 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.

INTENDED_EDIT: Move the original and horizontally flipped original views’ TTA weight halfway from the current best 1.0625 toward the accuracy-losing 1.125 setting.

EVIDENCE: Weight 1.0625 preserved 9,290 correct with 0.20246240730285645 cross-entropy, while 1.125 reduced cross-entropy to 0.2024369430541992 but lost two correct predictions; their midpoint directly probes the accuracy-preserving boundary.

<<<<<<< SEARCH
        view_weights = (1.0625, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.09375, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE