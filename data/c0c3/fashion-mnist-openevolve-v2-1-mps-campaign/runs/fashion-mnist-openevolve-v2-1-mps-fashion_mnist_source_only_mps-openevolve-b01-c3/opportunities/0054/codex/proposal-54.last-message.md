MECHANISM: Midpoint-reconstructed half-density tail averaging

HYPOTHESIS: Approximating omitted odd tail iterates by interpolation between adjacent sampled iterates will exceed the current 9,283 correct predictions and approach the dense average’s 9,285, without its per-step averaging cost.

INTENDED_EDIT: Anchor the average immediately before the final-10% window, retain every-other-iterate updates, and use quadrature weights that reproduce the dense linear-recency average when consecutive parameter trajectories are locally linear.

EVIDENCE: Dense final-10% linear averaging achieved 9,285 correct, while half-density linear averaging completed with 9,283; reconstructing the missing iterates directly targets that two-image gap with only one additional parameter copy.

<<<<<<< SEARCH
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
=======
    optimizer.tail_average_count = 0
    optimizer.tail_average_total_weight = 0.0
    optimizer.tail_average_parameters = [
>>>>>>> REPLACE

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
    tail_distance = total_steps - step - 1
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]

    if step + 1 == optimizer.tail_average_start - 1:
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.tail_average_parameters,
                parameters,
            ):
                average.copy_(parameter)
        optimizer.tail_average_total_weight = 0.5

    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 2 == 0
    ):
        optimizer.tail_average_count += 1
        sample_weight = 4.0 * optimizer.tail_average_count
        if step + 1 == total_steps:
            sample_weight -= optimizer.tail_average_count + 0.5
        optimizer.tail_average_total_weight += sample_weight
        average_weight = (
            sample_weight / optimizer.tail_average_total_weight
        )
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