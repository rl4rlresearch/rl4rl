MECHANISM: Temporally sparse linear tail averaging

HYPOTHESIS: Linearly averaging 20 evenly spaced iterates across the final 10% will exceed the current 9,281 correct predictions while reducing averaging overhead enough to finish verification.

INTENDED_EDIT: Restore the strongest final-10% linear-recency averaging distribution, but sample every fourth iterate so its effective temporal center is preserved with one quarter as many averaging updates.

EVIDENCE: Final-10% linear averaging achieved the best verified result of 9,285 correct, whereas the current final-15% sparse uniform average reached 9,281; repeated full-density restorations timed out, motivating a compute-reduced approximation.

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.85 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 3 != 2
    ):
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
=======
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 4 == 0
    ):
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
>>>>>>> REPLACE