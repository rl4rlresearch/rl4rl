MECHANISM: Tail exponential weight averaging

HYPOTHESIS: Replacing the final training iterate with a 0.99-decay EMA over the second half of training will exceed 9,249 correct predictions by reducing late optimization noise without changing the successful architecture or exposure budget.

INTENDED_EDIT: Track an exponential moving average of model parameters after halfway through training and install those averaged weights after the final optimizer step.

EVIDENCE: Optimization changes previously improved correctness while multiple architecture changes reduced it; tail averaging targets optimization stability while preserving the 245,044-parameter representation and batch size 96.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    optimizer._model_parameters = list(model.parameters())
    optimizer._ema_shadow = None
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
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (
            (progress - warmup_fraction) / (1.0 - warmup_fraction)
        )
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)

    if progress >= 0.5:
        parameters = optimizer._model_parameters
        with torch.no_grad():
            if optimizer._ema_shadow is None:
                optimizer._ema_shadow = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                for averaged, parameter in zip(
                    optimizer._ema_shadow, parameters
                ):
                    averaged.lerp_(parameter.detach(), 0.01)

            if step + 1 >= total_steps:
                for parameter, averaged in zip(
                    parameters, optimizer._ema_shadow
                ):
                    parameter.copy_(averaged)

    warmup_fraction = 0.05
    if progress < warmup_fraction:
        multiplier = 0.2 + 0.8 * progress / warmup_fraction
    else:
        decay_progress = (
            (progress - warmup_fraction) / (1.0 - warmup_fraction)
        )
        multiplier = 0.05 + 0.95 * 0.5 * (
            1.0 + math.cos(math.pi * decay_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = PEAK_LR * multiplier
>>>>>>> REPLACE