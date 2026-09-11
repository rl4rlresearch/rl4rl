MECHANISM: Intermediate-recency endpoint-sampled tail quadrature

HYPOTHESIS: Recency^1.25 tail weighting with uniform ten-view logit pooling will exceed 9,287 correct predictions by interpolating between linear weighting’s higher correct count and recency^1.5 weighting’s lower cross-entropy.

INTENDED_EDIT: Restore ten-view logit averaging and endpoint-sampled half-density parameter averaging, using recency^1.25 weights.

EVIDENCE: Linear endpoint quadrature with ten-view logit pooling achieved 9,287 correct, while recency^1.5 achieved 9,286 with slightly lower cross-entropy; an intermediate exponent tests whether modest later-iterate emphasis preserves the former’s decisions while improving margins.

<<<<<<< SEARCH
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        return view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_count = 0
=======
    optimizer.tail_average_weight_sum = 0.0
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
    tail_index = step + 2 - optimizer.tail_average_start
    tail_length = total_steps - optimizer.tail_average_start + 1
    if (
        step + 1 >= optimizer.tail_average_start
        and (tail_index == 1 or tail_index % 2 == 0)
    ):
        recency_power = 1.25
        if tail_index == 1:
            sample_weight = 1.0
        elif tail_index == 2:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )
        elif tail_index == tail_length:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
            )
        else:
            sample_weight = (
                tail_index ** recency_power
                + 0.5 * (tail_index - 1) ** recency_power
                + 0.5 * (tail_index + 1) ** recency_power
            )

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