MECHANISM: Center-prioritized geometric view pooling with endpoint-sampled tail quadrature

HYPOTHESIS: Weighting the unshifted flip pair twice as strongly as each shifted pair while retaining the 9,287-correct logit-pooling and quadrature design will exceed 9,287 correct predictions by reducing replicated-boundary artifacts without sacrificing translation robustness.

INTENDED_EDIT: Restore the strongest endpoint-reconstructed half-density parameter average and replace uniform ten-view logit averaging with a 2:1 center-to-shifted weighted logit mean.

EVIDENCE: Uniform ten-view logit pooling with endpoint-sampled quadrature achieved the best result, 9,287 correct; the earlier center-prioritized probability-pooling trial timed out, leaving center weighting under the stronger logit-pooling rule unmeasured.

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
        return (
            2.0 * view_logits[0].sum(dim=0)
            + view_logits[1:].sum(dim=(0, 1))
        ) / 12.0
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