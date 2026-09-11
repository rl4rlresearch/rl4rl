MECHANISM: Late-training exponential weight averaging

HYPOTHESIS: Averaging the final half of the optimization trajectory with decay 0.98 will exceed 9,245 correct predictions by reducing endpoint variance without changing the architecture or exposure budget.

INTENDED_EDIT: Preserve the verified training regimen while maintaining an EMA of learned parameters during the final 391 optimizer steps and installing the averaged parameters after the last step.

EVIDENCE: The current 249,754-parameter design gained only 9 correct predictions from its latest bottleneck and has slightly worse cross-entropy than the 241,274-parameter predecessor; with almost no parameter headroom remaining, stabilizing its late cosine-decayed trajectory is the most direct isolated opportunity.

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
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    calls = getattr(optimizer, "_ema_calls", 0) + 1
    optimizer._ema_calls = calls
    averaging_start = max(total_steps // 2, 1)

    with torch.no_grad():
        if calls == averaging_start:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        elif calls > averaging_start:
            for average, parameter in zip(
                optimizer._ema_parameters, parameters
            ):
                average.lerp_(parameter, 0.02)

        if calls >= total_steps and hasattr(optimizer, "_ema_parameters"):
            for parameter, average in zip(
                parameters, optimizer._ema_parameters
            ):
                parameter.copy_(average)
>>>>>>> REPLACE