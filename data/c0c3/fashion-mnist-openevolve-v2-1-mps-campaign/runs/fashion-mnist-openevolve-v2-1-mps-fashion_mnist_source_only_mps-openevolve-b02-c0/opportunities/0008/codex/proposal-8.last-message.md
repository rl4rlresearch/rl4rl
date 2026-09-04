MECHANISM: End-of-training exponential weight averaging

HYPOTHESIS: Averaging the successful batch-48 model’s parameter trajectory will exceed 9,146 correct predictions by reducing late-training optimizer noise without changing its proven architecture, exposure, or update count.

INTENDED_EDIT: Maintain a bias-controlled exponential moving average of trainable parameters after every optimizer step and install the averaged weights after the final step.

EVIDENCE: Batch size 48 achieved the best result at 9,146 correct, while further update densification at batch size 44 fell to 9,142 and widening the classifier fell to 9,138; this motivates preserving the best configuration and stabilizing its learned endpoint.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6.0e-4,
        betas=(0.9, 0.99),
        weight_decay=5.0e-4,
    )
    optimizer.ema_parameters = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    warmup_fraction = 0.05
    start_lr = 6.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.0e-4
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    warmup_fraction = 0.05
    start_lr = 6.0e-4
    peak_lr = 3.0e-3
    end_lr = 1.0e-4
    if progress < warmup_fraction:
        lr = start_lr + (peak_lr - start_lr) * progress / warmup_fraction
    else:
        decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        lr = end_lr + 0.5 * (peak_lr - end_lr) * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    ema_decay = min(0.995, (step + 1.0) / (step + 10.0))
    with torch.no_grad():
        for averaged, parameter in zip(optimizer.ema_parameters, parameters):
            averaged.lerp_(parameter, 1.0 - ema_decay)
        if step >= total_steps - 1:
            for parameter, averaged in zip(parameters, optimizer.ema_parameters):
                parameter.copy_(averaged)
>>>>>>> REPLACE