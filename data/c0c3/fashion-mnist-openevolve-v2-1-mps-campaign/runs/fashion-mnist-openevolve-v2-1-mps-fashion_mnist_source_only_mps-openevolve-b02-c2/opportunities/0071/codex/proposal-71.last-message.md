MECHANISM: Sparse final-half parameter EMA

HYPOTHESIS: A 0.99-per-step EMA sampled every eight steps will exceed 9,322 correct predictions by suppressing late-update noise while avoiding the timeout of the prior every-step EMA.

INTENDED_EDIT: Restore ordinary BatchNorm momentum and maintain a mathematically time-adjusted parameter EMA every eight steps during the final half of training, installing it after the last update.

EVIDENCE: Hard-maximum attention with ordinary BatchNorm finished in 75.3 seconds at 9,320 correct, while sample-weighted BatchNorm provided no accuracy gain; the prior every-step EMA timed out, motivating the same stabilization at one-eighth the update frequency.

<<<<<<< SEARCH
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
=======
    del step, total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    if step >= total_steps // 2:
        if not hasattr(optimizer, "_ema_parameters"):
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            optimizer._ema_step = step
        elif step % 8 == 0 or step >= total_steps:
            elapsed_steps = step - optimizer._ema_step
            decay = 0.99 ** elapsed_steps
            torch._foreach_lerp_(
                optimizer._ema_parameters,
                parameters,
                1.0 - decay,
            )
            optimizer._ema_step = step

        if step >= total_steps:
            torch._foreach_copy_(parameters, optimizer._ema_parameters)

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE