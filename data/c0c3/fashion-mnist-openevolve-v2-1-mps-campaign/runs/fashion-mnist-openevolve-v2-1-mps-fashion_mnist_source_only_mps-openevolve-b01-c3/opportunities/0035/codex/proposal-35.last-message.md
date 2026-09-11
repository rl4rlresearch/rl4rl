MECHANISM: Intermediate-recency tail parameter averaging

HYPOTHESIS: Weighting final-10% iterates proportional to their index^1.5 will exceed 9,285 correct predictions by increasing the beneficial recency bias of linear averaging while retaining the temporal coverage lost by final-5% averaging.

INTENDED_EDIT: Restore the proven final-10% averaging window and replace uniform averaging with exact online 1.5-power recency weighting.

EVIDENCE: Final-10% linear weighting achieved 9,285 correct versus 9,282 for uniform weighting, while final-5% averaging fell to 9,275; this motivates testing a stronger intermediate recency profile without shortening the successful window.

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.95 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_weight_sum = 0.0
    optimizer.tail_average_parameters = [
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
        parameters = [
=======
        optimizer.tail_average_count += 1
        iterate_weight = float(optimizer.tail_average_count) ** 1.5
        optimizer.tail_average_weight_sum += iterate_weight
        average_weight = iterate_weight / optimizer.tail_average_weight_sum
        parameters = [
>>>>>>> REPLACE