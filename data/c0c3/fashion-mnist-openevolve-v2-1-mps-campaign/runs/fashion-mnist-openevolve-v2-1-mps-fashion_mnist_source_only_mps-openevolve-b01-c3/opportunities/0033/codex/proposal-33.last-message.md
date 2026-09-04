MECHANISM: Quadratic-recency tail parameter averaging

HYPOTHESIS: Quadratically weighting later iterates within the final 10% window will exceed 9,285 correct predictions by further reducing BatchNorm-state mismatch while preserving the full successful averaging span.

INTENDED_EDIT: Replace linear recency weights with exact online weights proportional to the square of each tail iterate’s index.

EVIDENCE: Final-10% linear weighting achieved 9,285 correct versus 9,282 for uniform weighting, showing that recency bias improved the strongest baseline without extra training work; quadratic weighting directly tests whether a stronger smooth bias extends that gain.

<<<<<<< SEARCH
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
=======
        average_weight = (
            6.0 * optimizer.tail_average_count
            / (
                (optimizer.tail_average_count + 1.0)
                * (2.0 * optimizer.tail_average_count + 1.0)
            )
        )
>>>>>>> REPLACE