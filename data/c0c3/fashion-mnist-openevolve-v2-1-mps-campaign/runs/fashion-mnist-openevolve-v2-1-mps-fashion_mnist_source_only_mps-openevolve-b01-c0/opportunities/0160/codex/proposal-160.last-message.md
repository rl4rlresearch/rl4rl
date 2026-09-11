MECHANISM: Log-domain boundary bisection

HYPOTHESIS: Setting the unshifted-view weight to 1.0665515884757042 will preserve 9,290 correct predictions while reducing validation cross-entropy below 0.2024606979370117.

INTENDED_EDIT: Increase the original and horizontally flipped original views’ TTA weight to the midpoint between the highest tested accuracy-preserving value and the lowest tested value that lost one prediction.

EVIDENCE: Weight 1.0665515810251236 preserved 9,290 correct, while 1.0665515959262848 produced 9,289 with lower cross-entropy; their midpoint is the most informative remaining boundary probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515884757042, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE