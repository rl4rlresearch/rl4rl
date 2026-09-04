MECHANISM: Final-10% linearly recency-weighted parameter averaging

HYPOTHESIS: Restoring the verified final-10% linear tail average will exceed the current 9,281 correct predictions and recover approximately 9,285 correct.

INTENDED_EDIT: Replace the subsampled uniform final-15% average with averaging every final-10% iterate using exact linear recency weights.

EVIDENCE: Reference Design 2 achieved the strongest verified result—9,285 correct—while the current broader subsampled uniform average achieved 9,281.

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.85 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 3 != 2
    ):
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
=======
    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE