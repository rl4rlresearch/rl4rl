MECHANISM: Shorter-horizon full-state exponential weight consolidation

HYPOTHESIS: A 0.98-decay full-state EMA will exceed 9,343 correct predictions by tracking the final low-learning-rate solution more closely than the successful 0.985 EMA while retaining its averaging of BatchNorm statistics.

INTENDED_EDIT: Restore foreach-accelerated averaging of all floating-point parameters and normalization buffers, and shorten the EMA decay from 0.985 to 0.98.

EVIDENCE: Full-state 0.985 EMA achieved the best result at 9,343 correct, outperforming full-state 0.99 EMA at 9,334 and parameter-only 0.99 EMA at 9,331; this motivates one further localized shortening of the successful EMA horizon.

<<<<<<< SEARCH
    optimizer._averaging_model = model
    optimizer._averaged_parameters = None
    optimizer._averaging_last_step = None
=======
    optimizer._averaging_values = tuple(
        value
        for value in model.state_dict().values()
        if torch.is_floating_point(value)
    )
    optimizer._averaged_values = None
    optimizer._averaging_last_step = None
>>>>>>> REPLACE

<<<<<<< SEARCH
    if should_average:
        parameters = tuple(optimizer._averaging_model.parameters())
        averaged_parameters = optimizer._averaged_parameters
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            if averaged_parameters is None:
                averaged_parameters = [
                    parameter.detach().clone()
                    for parameter in parameters
                ]
            else:
                elapsed = max(step - last_step, 1)
                decay = 0.99 ** elapsed
                torch._foreach_mul_(averaged_parameters, decay)
                torch._foreach_add_(
                    averaged_parameters,
                    parameters,
                    alpha=1.0 - decay,
                )
        optimizer._averaged_parameters = averaged_parameters
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(
                    parameters,
                    averaged_parameters,
                )
=======
    if should_average:
        values = optimizer._averaging_values
        averaged_values = optimizer._averaged_values
        last_step = optimizer._averaging_last_step
        with torch.no_grad():
            if averaged_values is None:
                averaged_values = [
                    value.detach().clone() for value in values
                ]
            else:
                elapsed = max(step - last_step, 1)
                decay = 0.98 ** elapsed
                torch._foreach_mul_(averaged_values, decay)
                torch._foreach_add_(
                    averaged_values,
                    values,
                    alpha=1.0 - decay,
                )
        optimizer._averaged_values = averaged_values
        optimizer._averaging_last_step = step

        if step >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(values, averaged_values)
>>>>>>> REPLACE