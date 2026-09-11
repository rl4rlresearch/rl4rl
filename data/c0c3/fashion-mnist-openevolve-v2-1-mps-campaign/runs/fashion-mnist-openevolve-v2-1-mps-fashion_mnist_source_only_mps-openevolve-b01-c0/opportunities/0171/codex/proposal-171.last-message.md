MECHANISM: Log-domain accuracy-boundary retry

HYPOTHESIS: An unshifted-view weight of 1.0665515868386138 will preserve 9,290 correct predictions and may reduce validation cross-entropy below 0.2024606979370117.

INTENDED_EDIT: Set the original and horizontally flipped original views’ shared TTA weight to the unresolved midpoint of the tightest known accuracy boundary.

EVIDENCE: Weight 1.0665515868313378 preserved 9,290 predictions, while 1.0665515868458897 produced 9,289. Their midpoint was not evaluated because verification timed out; otherwise identical runs completed in 49.7–72.3 seconds, so retrying it remains the most informative probe.

<<<<<<< SEARCH
        view_weights = (1.0665515661239624, 1.0, 1.0, 1.0, 1.0)
=======
        view_weights = (1.0665515868386138, 1.0, 1.0, 1.0, 1.0)
>>>>>>> REPLACE