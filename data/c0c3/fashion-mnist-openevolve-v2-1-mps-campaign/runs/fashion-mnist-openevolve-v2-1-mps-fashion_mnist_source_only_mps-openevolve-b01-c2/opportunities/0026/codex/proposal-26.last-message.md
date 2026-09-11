MECHANISM: Low-overhead tail weight averaging

HYPOTHESIS: Averaging parameter iterates over the final 10% of the winning linear curriculum will exceed 9,330 correct predictions by reducing late optimization noise without the full-run overhead that caused EMA verification to time out.

INTENDED_EDIT: Restore static 0.02 label smoothing and uniformly average only the final 10% of learned parameter states, installing the average before validation.

EVIDENCE: Linear ensemble annealing with static 0.02 smoothing achieved the best result of 9,330 correct; terminal smoothing decay fell to 9,322, while full-run EMA timed out, motivating a cheaper tail-only stabilization test.

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02 * min(1.0, 4.0 * (1.0 - progress))
=======
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.5 + 0.5 * progress
    label_smoothing = 0.02
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier

    average_start = int(0.9 * total_steps)
    if step >= average_start:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                count = optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average,
                    parameters,
                ):
                    average.add_(
                        parameter.detach() - average,
                        alpha=1.0 / (count + 1),
                    )
                optimizer._tail_average_count = count + 1

            if step + 1 >= total_steps:
                for parameter, average in zip(
                    parameters,
                    optimizer._tail_average,
                ):
                    parameter.copy_(average)
>>>>>>> REPLACE