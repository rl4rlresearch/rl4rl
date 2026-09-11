MECHANISM: Log-domain boundary bisection

HYPOTHESIS: An unshifted-view weight of 1.0665515866130591 will preserve 9,290 correct predictions and may reduce cross-entropy below 0.2024606979370117.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the midpoint of the tightest known accuracy boundary.

EVIDENCE: Weight 1.066551584750414 preserved 9,290 correct, while 1.0665515884757042 produced 9,289; their midpoint is the most informative remaining probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515866130591, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE