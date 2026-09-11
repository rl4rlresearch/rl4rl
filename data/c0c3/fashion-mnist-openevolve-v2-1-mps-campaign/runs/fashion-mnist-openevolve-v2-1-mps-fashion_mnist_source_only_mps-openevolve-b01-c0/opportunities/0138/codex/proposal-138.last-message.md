MECHANISM: Canonical-view weight boundary bisection

HYPOTHESIS: Setting each unshifted view’s weight to 1.0703125 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.20246240730285645.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight from 1.0625× to 1.0703125×, leaving all shifted views unchanged.

EVIDENCE: Weight 1.0625 retained 9,290 correct with 0.20246240730285645 cross-entropy, while 1.078125 lost one correct but lowered cross-entropy to 0.2024558723449707; their midpoint is the most informative probe of the accuracy-preserving boundary.

<<<<<<< SEARCH
        view_weights = (1.0625, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0703125, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE