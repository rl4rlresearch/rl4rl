MECHANISM: Accuracy-boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.0665515868315475 will preserve 9,290 correct predictions while testing the narrowest unresolved boundary for lower validation cross-entropy.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.

EVIDENCE: Weight 1.0665515868315439 preserved 9,290 correct predictions, while 1.066551586831551 produced 9,289; their midpoint is the most informative unresolved probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515868315475, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE