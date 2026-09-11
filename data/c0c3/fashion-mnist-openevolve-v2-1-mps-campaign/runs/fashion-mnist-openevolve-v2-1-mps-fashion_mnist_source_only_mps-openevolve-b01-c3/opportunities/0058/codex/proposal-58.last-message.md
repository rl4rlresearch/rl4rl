MECHANISM: Fused dense linear-recency tail averaging

HYPOTHESIS: Fusing all parameter-average updates into multi-tensor operations will make dense final-10% averaging finish within the time limit and recover the reference result of at least 9,285 correct predictions.

INTENDED_EDIT: Average every final-window iterate as in the strongest reference design, replacing per-parameter Python loops with fused foreach lerp and copy operations.

EVIDENCE: Dense final-10% linear averaging achieved the best verified result of 9,285 correct, while half-density averaging reached 9,283; later dense variants timed out, making averaging overhead the targeted constraint.

<<<<<<< SEARCH
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
=======
    if step + 1 >= optimizer.tail_average_start:
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            torch._foreach_lerp_(
                optimizer.tail_average_parameters,
                parameters,
                average_weight,
            )

            if step + 1 == total_steps:
                torch._foreach_copy_(
                    parameters,
                    optimizer.tail_average_parameters,
                )
>>>>>>> REPLACE