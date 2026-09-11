MECHANISM: Log-domain accuracy-boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.0665515868322473 will preserve 9,290 correct predictions while probing for lower validation cross-entropy.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.

EVIDENCE: Weight 1.0665515868313378 preserved 9,290 predictions, while 1.0665515868331568 produced 9,289; their midpoint is the most informative unresolved probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515868322473, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE