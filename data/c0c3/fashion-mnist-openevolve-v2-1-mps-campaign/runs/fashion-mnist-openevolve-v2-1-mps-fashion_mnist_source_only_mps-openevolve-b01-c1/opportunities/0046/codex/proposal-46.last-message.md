MECHANISM: Fused coarse-checkpoint parameter EMA

HYPOTHESIS: A foreach-fused 0.99-decay EMA sampled every 32 steps will exceed 9,239 correct predictions by suppressing endpoint noise without the runtime overhead that caused per-step EMA to time out.

INTENDED_EDIT: Maintain a fused parameter EMA at 32-step intervals, account for elapsed steps in its decay, and install the averaged parameters after the final optimizer step.

EVIDENCE: The cosine-curriculum model remains best at 9,239 correct, while architecture, head, loss, and augmentation changes failed to improve it; per-step EMA timed out, so substantially reducing update frequency and kernel dispatch isolates its still-unresolved optimization benefit.

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier

    completed_steps = step + 1
    if completed_steps % 32 == 0 or completed_steps == total_steps:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_ema_parameters"):
                optimizer._ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_step = completed_steps
            else:
                elapsed_steps = completed_steps - optimizer._ema_step
                decay = 0.99 ** elapsed_steps
                torch._foreach_mul_(optimizer._ema_parameters, decay)
                torch._foreach_add_(
                    optimizer._ema_parameters,
                    parameters,
                    alpha=1.0 - decay,
                )
                optimizer._ema_step = completed_steps
            if completed_steps == total_steps:
                for parameter, averaged_parameter in zip(
                    parameters, optimizer._ema_parameters
                ):
                    parameter.copy_(averaged_parameter)
>>>>>>> REPLACE