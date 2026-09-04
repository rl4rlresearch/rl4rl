MECHANISM: Sparse tail exponential parameter averaging

HYPOTHESIS: Updating an EMA every 32 steps during the latter half of training will exceed 9,280 correct predictions while finishing within the time limit by smoothing batch-48 optimization with roughly 16× fewer averaging updates than the timed-out every-two-step implementation.

INTENDED_EDIT: Maintain a 0.85-decay parameter EMA at 32-step intervals during the second half of training and install it after the final optimizer step.

EVIDENCE: Batch size 48 achieved the best result at 9,280 correct, while denser tail averaging timed out; sparsifying the same averaging mechanism directly targets its observed runtime cost.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
=======
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    update_number = step + 1
    update_average = (
        progress >= 0.5
        and (update_number % 32 == 0 or update_number == total_steps)
    )
    if update_average:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            averages = getattr(optimizer, "_tail_ema", None)
            if averages is None:
                averages = [
                    parameter.detach().clone() for parameter in parameters
                ]
                setattr(optimizer, "_tail_ema", averages)
            else:
                for average, parameter in zip(averages, parameters):
                    average.lerp_(parameter.detach(), 0.15)

            if update_number == total_steps:
                for parameter, average in zip(parameters, averages):
                    parameter.copy_(average)
>>>>>>> REPLACE