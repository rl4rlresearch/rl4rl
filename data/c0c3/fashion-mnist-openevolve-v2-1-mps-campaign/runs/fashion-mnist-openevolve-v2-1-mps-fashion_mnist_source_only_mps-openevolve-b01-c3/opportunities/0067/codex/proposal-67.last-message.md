MECHANISM: Three-halves-recency endpoint-sampled tail quadrature

HYPOTHESIS: Weighting tail iterates by recency^1.5 while retaining uniform ten-view logit pooling will exceed 9,287 correct predictions by reducing mismatch with terminal BatchNorm statistics without discarding the smoothing benefit of the final-10% window.

INTENDED_EDIT: Restore the strongest ten-view logit pooling and endpoint-sampled quadrature, but reconstruct a moderately later-biased recency^1.5 parameter average instead of the linear-recency average.

EVIDENCE: Endpoint-sampled linear quadrature reached 9,286 correct and uniform ten-view logit pooling raised it to 9,287; BatchNorm-statistic experiments indicate parameter/statistic alignment remains unresolved, motivating a computation-neutral shift toward later tail iterates.

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
    optimizer.tail_average_count = 0
=======
    optimizer.tail_average_weight_sum = 0.0
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
        recency_power = 1.5
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