MECHANISM: Runtime-headroom-funded exact parameter EMA

HYPOTHESIS: Applying the demonstrated 0.995 per-step parameter EMA to the completed batch-64 design will finish within the time limit and exceed 9,119 correct predictions by suppressing late-update noise.

INTENDED_EDIT: Preserve the current architecture, training, augmentation, schedule, and ten-view inference; maintain a fused EMA of all learned parameters and install it after the final optimizer update.

EVIDENCE: Exact 0.995 EMA previously raised the strongest batch-32 result from 9,214 to 9,241 correct with about 1.6 seconds of added runtime, while the current design completed in 63.63 seconds and therefore has substantially more headroom.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
    optimizer._ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
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
    ]
    ema_parameters = optimizer._ema_parameters
    with torch.no_grad():
        torch._foreach_lerp_(ema_parameters, parameters, 0.005)
        if step + 1 >= total_steps:
            torch._foreach_copy_(parameters, ema_parameters)

    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE