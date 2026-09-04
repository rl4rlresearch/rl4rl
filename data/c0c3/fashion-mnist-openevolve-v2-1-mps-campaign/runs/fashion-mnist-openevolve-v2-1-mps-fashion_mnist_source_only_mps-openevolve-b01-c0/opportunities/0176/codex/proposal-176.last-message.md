MECHANISM: Accuracy-boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.0665515868317925 will preserve 9,290 correct predictions while testing the narrowest unresolved accuracy boundary.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known boundary.

EVIDENCE: Weight 1.0665515868313378 preserved 9,290 correct predictions, while 1.0665515868322473 produced 9,289; their midpoint is the most informative next probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515868317925, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE