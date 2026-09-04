MECHANISM: Endpoint-sampled linear-tail quadrature

HYPOTHESIS: Reconstructing the dense linear-recency average from the first and every second tail iterate will recover at least 9,285 correct predictions while retaining approximately half-density averaging cost.

INTENDED_EDIT: Restore proven default BatchNorm momentum and replace the biased half-density average with online quadrature weights that exactly reproduce dense averaging when intervening parameter trajectories are locally linear.

EVIDENCE: Dense tail averaging achieved 9,285 correct versus 9,283 for half-density averaging, while momentum 0.04 reduced the latter to 9,280. The prior midpoint reconstruction timed out after adding another parameter copy; sampling the first tail iterate permits the same reconstruction using the existing average buffer.

<<<<<<< SEARCH
def build_model() -> nn.Module:
    model = ImageClassifier()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.04
    return model
=======
def build_model() -> nn.Module:
    return ImageClassifier()
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_count = 0
    optimizer.tail_average_parameters = [
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_weight_sum = 0.0
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
    tail_index = step + 2 - optimizer.tail_average_start
    tail_length = total_steps - optimizer.tail_average_start + 1
    if (
        step + 1 >= optimizer.tail_average_start
        and (tail_index == 1 or tail_index % 2 == 0)
    ):
        if tail_index == 1:
            sample_weight = 1.0
        elif tail_index == 2:
            sample_weight = 3.5
        elif tail_index == tail_length:
            sample_weight = tail_index + 0.5 * (tail_index - 1)
        else:
            sample_weight = 2.0 * tail_index

        optimizer.tail_average_weight_sum += sample_weight
        average_weight = sample_weight / optimizer.tail_average_weight_sum
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