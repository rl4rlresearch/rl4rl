MECHANISM: Sparse late-checkpoint weight averaging

HYPOTHESIS: Averaging evenly spaced parameter snapshots from the final 10% of the cosine-decayed trajectory will exceed 9,249 correct predictions by reducing terminal mini-batch noise without adding model parameters or inference work.

INTENDED_EDIT: Accumulate every fourth low-learning-rate checkpoint during the final 10% of training and replace the final parameters with their arithmetic average.

EVIDENCE: The best design reached 9,249 correct, while added capacity and altered inference aggregation repeatedly timed out; this isolates checkpoint stability with negligible computation and preserves the successful architecture, augmentation, and ensemble.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = lr
=======
    is_final_step = step + 1 >= total_steps
    if progress >= 0.90 and (step % 4 == 0 or is_final_step):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        average_count = getattr(optimizer, "_late_average_count", 0)
        with torch.no_grad():
            if average_count == 0:
                optimizer._late_averaged_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                update_weight = 1.0 / (average_count + 1)
                for average, parameter in zip(
                    optimizer._late_averaged_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), update_weight)
            optimizer._late_average_count = average_count + 1

            if is_final_step:
                for parameter, average in zip(
                    parameters, optimizer._late_averaged_parameters
                ):
                    parameter.copy_(average)

    for group in optimizer.param_groups:
        group["lr"] = lr
>>>>>>> REPLACE