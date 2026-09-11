MECHANISM: Tail exponential parameter averaging

HYPOTHESIS: Averaging learned parameters over the latter half of training will exceed 9,280 correct predictions by reducing variance from batch-48 optimization without requiring the slower batch-40 update count.

INTENDED_EDIT: Maintain a low-overhead EMA every two optimizer steps during the second half of training, then install the averaged parameters after the final step.

EVIDENCE: Batch 48 improved validation_correct from 9,238 to 9,280, while batch 40 timed out; parameter averaging tests whether the benefits of smoother optimization can be captured within the proven runtime regime.

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

    if progress >= 0.5 and (
        (step + 1) % 2 == 0 or step + 1 == total_steps
    ):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_ema"):
                optimizer._tail_ema = [
                    parameter.detach().clone() for parameter in parameters
                ]
            else:
                for average, parameter in zip(
                    optimizer._tail_ema, parameters
                ):
                    average.lerp_(parameter.detach(), 0.01)

            if step + 1 == total_steps:
                for parameter, average in zip(
                    parameters, optimizer._tail_ema
                ):
                    parameter.copy_(average)

    warmup = 0.08
    if progress < warmup:
        learning_rate = 8e-4 + (3.0e-3 - 8e-4) * progress / warmup
    else:
        decay = (progress - warmup) / (1.0 - warmup)
        multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * decay))
        learning_rate = 3.0e-3 * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
>>>>>>> REPLACE