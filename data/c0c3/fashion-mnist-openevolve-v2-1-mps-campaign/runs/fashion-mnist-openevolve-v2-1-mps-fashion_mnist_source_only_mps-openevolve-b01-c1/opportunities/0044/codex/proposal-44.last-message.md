MECHANISM: Exponential moving-average endpoint weights

HYPOTHESIS: A 0.99-decay parameter EMA will exceed 9,239 correct predictions by suppressing late-stage optimizer noise while preserving the proven model, objective, augmentation, and training trajectory.

INTENDED_EDIT: Track an EMA of trainable parameters after every optimizer step and install the averaged weights after the final step for validation.

EVIDENCE: The cosine-curriculum design achieved the best result at 9,239 correct, while subsequent architecture, head, loss-schedule, and augmentation changes regressed or timed out; endpoint weight averaging tests a compute-light optimization mechanism without disturbing that successful design.

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
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
    optimizer._model_parameters = parameters
    optimizer._ema_parameters = [
        parameter.detach().clone() for parameter in parameters
    ]
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
    model_parameters = optimizer._model_parameters
    ema_parameters = optimizer._ema_parameters
    with torch.no_grad():
        for averaged, parameter in zip(ema_parameters, model_parameters):
            averaged.lerp_(parameter, 0.01)
        if step + 1 >= total_steps:
            for parameter, averaged in zip(model_parameters, ema_parameters):
                parameter.copy_(averaged)

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
>>>>>>> REPLACE