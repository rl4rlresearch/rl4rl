MECHANISM: Hybrid-density linear tail averaging

HYPOTHESIS: Densely averaging only the classifier while averaging the backbone every fourth iterate will exceed 9,283 correct predictions by recovering decision-boundary stability from the 9,285-correct dense average without its full runtime cost.

INTENDED_EDIT: Keep the proven sparse final-10% average for all parameters, add a separate dense linear-recency average for the small classifier, and install the dense classifier average after the sparse full-model average.

EVIDENCE: Increasing final-10% averaging density improved correctness monotonically from 9,281 every fourth iterate, to 9,283 every second iterate, to 9,285 every iterate; concentrating dense averaging on the classifier targets prediction-changing parameters with much less overhead.

<<<<<<< SEARCH
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
=======
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.head_average_count = 0
    optimizer.head_average_sources = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    optimizer.head_average_parameters = [
        parameter.detach().clone()
        for parameter in optimizer.head_average_sources
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    tail_distance = total_steps - step - 1
    if (
        step + 1 >= optimizer.tail_average_start
        and tail_distance % 4 == 0
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
        optimizer.head_average_count += 1
        head_weight = 2.0 / (optimizer.head_average_count + 1.0)
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.head_average_parameters,
                optimizer.head_average_sources,
            ):
                average.lerp_(parameter, head_weight)

        tail_distance = total_steps - step - 1
        if tail_distance % 4 == 0:
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
                    for parameter, average in zip(
                        optimizer.head_average_sources,
                        optimizer.head_average_parameters,
                    ):
                        parameter.copy_(average)
>>>>>>> REPLACE