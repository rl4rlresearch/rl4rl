MECHANISM: Square-root-recency endpoint-sampled tail quadrature

HYPOTHESIS: Recency^0.5 tail weighting will exceed 9,287 correct predictions by extending the observed accuracy-favoring shift toward earlier tail iterates while retaining uniform ten-view logit pooling.

INTENDED_EDIT: Restore uniform ten-view logit averaging and endpoint-sampled half-density parameter averaging, then reduce the recency exponent from the best-tested 0.75 to 0.5.

EVIDENCE: Linear weighting achieved 9,287 correct, recency^1.25 and ^1.5 fell to 9,286, and recency^0.75 recovered 9,287; testing 0.5 continues the supported direction without adding parameters or computation.

<<<<<<< SEARCH
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        position_logits = view_logits.mean(dim=1)
        return torch.logsumexp(
            F.log_softmax(position_logits, dim=-1), dim=0
        ) - math.log(position_logits.shape[0])
=======
        view_logits = self._predict(torch.cat(views, dim=0)).reshape(
            5, 2, images.shape[0], 10
        )
        return view_logits.mean(dim=(0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_count = 0
=======
    optimizer.tail_average_start = max(1, int(0.90 * total_steps))
    optimizer.tail_average_weight_sum = 0.0
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        recency_power = 0.5
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