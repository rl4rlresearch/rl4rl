MECHANISM: Exponential moving-average weight consolidation

HYPOTHESIS: Replacing the final noisy AdamW weights with a 0.99-decay exponential moving average will exceed 9,319 correct predictions by stabilizing the converged classifier without adding regularization during optimization.

INTENDED_EDIT: Track an EMA of every learned parameter after each optimizer step and copy the averaged weights into the model after the final step.

EVIDENCE: Fine-grained changes to dropout, label smoothing, and learning-rate scheduling all failed to improve the 9,319-correct configuration, motivating an orthogonal endpoint-averaging change that preserves its successful training trajectory.

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
    else:
        cosine_progress = (progress - 0.05) / 0.95
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    if progress < 0.05:
        multiplier = 0.2 + 0.8 * progress / 0.05
    else:
        cosine_progress = (progress - 0.05) / 0.95
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    ema_step = getattr(optimizer, "_ema_step", 0) + 1
    optimizer._ema_step = ema_step
    with torch.no_grad():
        if not hasattr(optimizer, "_ema_parameters"):
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            torch._foreach_lerp_(
                optimizer._ema_parameters,
                parameters,
                0.01,
            )
        if ema_step == total_steps:
            for parameter, averaged in zip(
                parameters,
                optimizer._ema_parameters,
            ):
                parameter.copy_(averaged)
>>>>>>> REPLACE