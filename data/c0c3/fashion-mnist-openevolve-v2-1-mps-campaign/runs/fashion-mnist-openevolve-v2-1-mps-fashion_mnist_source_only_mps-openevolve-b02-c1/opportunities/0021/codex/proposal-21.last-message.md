MECHANISM: Conservative tail-EMA weight interpolation

HYPOTHESIS: Blending 25% of the sparse tail EMA into the final weights will retain at least 9,280 correct predictions while lowering validation cross-entropy, improving validation_score.

INTENDED_EDIT: Sample a parameter EMA every 32 steps during the second half of training, then interpolate it 25% into the final parameters instead of fully replacing them.

EVIDENCE: Full sparse tail EMA finished within the time limit and reduced cross-entropy from 0.19808 to 0.19631 but lost one correct prediction; a conservative interpolation should capture some smoothing benefit while staying closer to the higher-accuracy final weights.

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup = 0.08
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup = 0.08
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    update = step + 1
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if update >= total_steps // 2 and update % 32 == 0:
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_ema"):
                optimizer._tail_ema = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                for average, parameter in zip(
                    optimizer._tail_ema, parameters
                ):
                    average.mul_(0.85).add_(parameter, alpha=0.15)

    if update == total_steps and hasattr(optimizer, "_tail_ema"):
        with torch.no_grad():
            for parameter, average in zip(
                parameters, optimizer._tail_ema
            ):
                parameter.lerp_(average, 0.25)
>>>>>>> REPLACE