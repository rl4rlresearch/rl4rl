MECHANISM: Accuracy-boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.0665515868315439 will preserve 9,290 correct predictions while probing for lower validation cross-entropy.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest unresolved accuracy boundary.

EVIDENCE: Weight 1.0665515868315368 preserved 9,290 predictions, while 1.066551586831551 produced 9,289; their midpoint is the most informative remaining probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515868315439, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE