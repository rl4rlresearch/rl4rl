MECHANISM: Dense linear-recency tail averaging

HYPOTHESIS: Averaging every iterate in the final 10% will reproduce at least 9,285 correct predictions, matching the strongest qualified design.

INTENDED_EDIT: Replace every-fourth-iterate tail averaging with dense per-iterate averaging while preserving the proven averaging window and linear-recency weights.

EVIDENCE: Reference Design 2 used this exact dense schedule, completed in 59.96 seconds, and achieved the best qualified result of 9,285 correct versus 9,281 for the current sparse schedule.

<<<<<<< SEARCH
    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 4 == 0
    ):
=======
    if step + 1 >= optimizer.tail_average_start:
>>>>>>> REPLACE