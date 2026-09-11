MECHANISM: Moment-matched extended-tail averaging

HYPOTHESIS: Quadratically weighting the sampled final-15% iterates will exceed 9,285 correct predictions by retaining broader trajectory diversity while shifting the average model’s effective time toward that of the successful final-10% linear average.

INTENDED_EDIT: Keep the current 80-update subsampled final-15% window, but replace uniform averaging with exact online weights proportional to each sampled iterate index squared.

EVIDENCE: Final-10% linear recency weighting achieved 9,285 correct versus 9,282 for uniform weighting, while the subsampled final-15% uniform average reached 9,281; quadratic weighting directly adds the proven recency preference to the broader window without increasing averaging work.

<<<<<<< SEARCH
        average_weight = 1.0 / optimizer.tail_average_count
=======
        count = optimizer.tail_average_count
        average_weight = 6.0 * count / (
            (count + 1.0) * (2.0 * count + 1.0)
        )
>>>>>>> REPLACE