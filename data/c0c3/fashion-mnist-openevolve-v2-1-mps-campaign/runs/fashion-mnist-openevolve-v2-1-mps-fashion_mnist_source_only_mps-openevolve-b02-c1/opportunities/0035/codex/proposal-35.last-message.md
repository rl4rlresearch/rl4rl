MECHANISM: Half-strength sparse tail parameter averaging

HYPOTHESIS: Interpolating the final parameters halfway toward a sparse mean of the last 20% of training will exceed 9,280 correct predictions while retaining some of the observed cross-entropy benefit of tail averaging.

INTENDED_EDIT: Sample parameters every eight optimizer steps during the final 20% of training, maintain their online mean, and blend the final parameters 50:50 with that mean.

EVIDENCE: Sparse tail EMA lowered validation cross-entropy from 0.19808 to 0.19631 while losing only one correct prediction, suggesting useful late-trajectory smoothing; partial interpolation should preserve more of the final iterate’s decision boundaries than full averaging.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
=======
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    update = step + 1
    in_tail = update >= int(0.8 * total_steps)
    sample_tail = in_tail and (update % 8 == 0 or update == total_steps)
    if sample_tail:
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
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average, parameters
                ):
                    average.lerp_(parameter.detach(), weight)

            if update == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_average
                ):
                    parameter.lerp_(average, 0.5)
>>>>>>> REPLACE