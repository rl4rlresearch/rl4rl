MECHANISM: Half-density linear-recency tail averaging

HYPOTHESIS: Linearly averaging every other iterate across the final 10% will exceed 9,281 correct predictions by recovering more of the dense average’s 9,285-correct benefit while retaining substantially lower averaging overhead.

INTENDED_EDIT: Extend averaging from the final 5% to the final 10%, sample every second iterate, and weight sampled iterates linearly by recency.

EVIDENCE: Dense final-10% linear averaging achieved 9,285 correct, while every-fourth-iterate averaging achieved 9,281 and completed verification; half-density sampling directly tests the accuracy–runtime midpoint.

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.95 * total_steps))
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 1.0 / optimizer.tail_average_count
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
=======
    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 2 == 0
    ):
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.lerp_(parameter, average_weight)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer.tail_average_parameters,
                ):
                    parameter.copy_(average)
>>>>>>> REPLACE