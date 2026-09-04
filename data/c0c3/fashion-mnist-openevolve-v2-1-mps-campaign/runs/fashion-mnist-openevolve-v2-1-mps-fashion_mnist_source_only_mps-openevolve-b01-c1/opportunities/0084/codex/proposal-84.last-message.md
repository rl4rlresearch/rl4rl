MECHANISM: Late-training parameter exponential moving average

HYPOTHESIS: Averaging model parameters over the final half of training will exceed 9,247 correct predictions by reducing optimizer noise while preserving the proven architecture, augmentation, and supervision schedules.

INTENDED_EDIT: Maintain a 0.99-decay EMA after training reaches 50% progress and install the averaged parameters after the final optimizer step for validation.

EVIDENCE: Recent changes to crop sampling, crop weighting, pair-loss allocation, ensemble allocation, and flip consistency all failed to improve the 9,247-correct design, motivating stabilization of its existing solution rather than another change to its learned invariances or objectives.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
    optimizer.ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    optimizer.ema_started = False
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if progress >= 0.5:
        with torch.no_grad():
            if not optimizer.ema_started:
                for average, parameter in zip(
                    optimizer.ema_parameters, parameters
                ):
                    average.copy_(parameter)
                optimizer.ema_started = True
            else:
                for average, parameter in zip(
                    optimizer.ema_parameters, parameters
                ):
                    average.lerp_(parameter, 0.01)
            if step + 1 >= total_steps:
                for parameter, average in zip(
                    parameters, optimizer.ema_parameters
                ):
                    parameter.copy_(average)

    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (
            progress - warmup_fraction
        ) / (1.0 - warmup_fraction)
        multiplier = 0.03 + 0.97 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
>>>>>>> REPLACE