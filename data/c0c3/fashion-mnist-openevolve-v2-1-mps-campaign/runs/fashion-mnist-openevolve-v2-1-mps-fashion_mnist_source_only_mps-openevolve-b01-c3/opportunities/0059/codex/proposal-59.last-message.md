MECHANISM: Contiguous-buffer dense tail averaging

HYPOTHESIS: Dense linear-recency averaging over the final 10% will reproduce at least 9,285 correct predictions while completing on time when each iterate is flattened and averaged with one contiguous tensor operation.

INTENDED_EDIT: Replace sparse per-parameter averaging with dense final-window averaging in a single flat parameter buffer, then install that buffer at the final step.

EVIDENCE: Dense final-10% averaging achieved the best verified result of 9,285 correct versus 9,281 for every-fourth-step averaging; the later fused multi-tensor attempt timed out, motivating a single contiguous-buffer implementation with fewer averaging dispatches.

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
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer.tail_average_vector = nn.utils.parameters_to_vector(
        optimizer.tail_average_parameters
    ).detach().clone()
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
        optimizer.tail_average_count += 1
        average_weight = 2.0 / (optimizer.tail_average_count + 1.0)
        with torch.no_grad():
            current_vector = nn.utils.parameters_to_vector(
                optimizer.tail_average_parameters
            )
            optimizer.tail_average_vector.lerp_(
                current_vector,
                average_weight,
            )

            if step + 1 == total_steps:
                nn.utils.vector_to_parameters(
                    optimizer.tail_average_vector,
                    optimizer.tail_average_parameters,
                )
>>>>>>> REPLACE