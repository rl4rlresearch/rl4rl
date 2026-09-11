MECHANISM: Bias-corrected exponential moving average of learned weights

HYPOTHESIS: Evaluating a short-horizon EMA of the noisy batch-64 training trajectory will exceed 9,122 correct predictions while lowering validation cross-entropy.

INTENDED_EDIT: Track an adaptive 0.99-decay EMA after every optimizer update and copy the averaged weights into the model after the final fixed-budget step.

EVIDENCE: Per-example stochastic translations raised accuracy from 9,073 to 9,082, and differential features raised it further to 9,122; averaging the resulting stochastic weight trajectory is an orthogonal way to retain these gains while reducing optimizer and augmentation variance.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay = [parameter for parameter in model.parameters() if parameter.ndim > 1]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim <= 1]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    model_parameters = list(model.parameters())
    decay = [parameter for parameter in model_parameters if parameter.ndim > 1]
    no_decay = [parameter for parameter in model_parameters if parameter.ndim <= 1]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
    optimizer.ema_model_parameters = model_parameters
    optimizer.ema_parameters = [
        parameter.detach().clone() for parameter in model_parameters
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
    warmup_fraction = 0.06
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        multiplier = 0.08 + 0.92 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.06
    if progress < warmup_fraction:
        multiplier = 0.1 + 0.9 * progress / warmup_fraction
    else:
        cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        multiplier = 0.08 + 0.92 * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = 1.25e-3 * multiplier

    ema_decay = min(0.99, (step + 1.0) / (step + 10.0))
    with torch.no_grad():
        for parameter, average in zip(
            optimizer.ema_model_parameters,
            optimizer.ema_parameters,
        ):
            average.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        if step + 1 >= total_steps:
            for parameter, average in zip(
                optimizer.ema_model_parameters,
                optimizer.ema_parameters,
            ):
                parameter.copy_(average)
>>>>>>> REPLACE