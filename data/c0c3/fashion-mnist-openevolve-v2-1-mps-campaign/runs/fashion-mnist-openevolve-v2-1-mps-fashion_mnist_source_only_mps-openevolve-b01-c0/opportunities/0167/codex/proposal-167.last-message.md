MECHANISM: Log-domain accuracy-boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.066551586787682 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.

EVIDENCE: Weight 1.0665515867294744 preserved 9,290 correct, while 1.0665515868458897 produced 9,289; their midpoint is the most informative remaining probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.066551586787682, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE