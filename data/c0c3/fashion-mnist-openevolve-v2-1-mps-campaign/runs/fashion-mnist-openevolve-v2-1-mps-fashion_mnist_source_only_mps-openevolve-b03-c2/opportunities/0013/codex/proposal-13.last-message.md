MECHANISM: Throughput-funded exact parameter EMA

HYPOTHESIS: Increasing batch size from 32 to 40 will provide enough training and validation throughput for the proven 0.995-decay EMA to finish verification and exceed 9,214 correct predictions.

INTENDED_EDIT: Use an evenly dividing batch size of 40 and maintain a fused per-step EMA of all learned parameters, installing the averaged parameters after the final update.

EVIDENCE: Per-step EMA produced 9,241 correct predictions versus the current 9,214 but missed the time limit at 82.89 seconds; batch 40 reduces optimizer steps from 3,126 to 2,500 and eliminates partial batches while preserving substantially denser optimization than batch 64.

<<<<<<< SEARCH
BATCH_SIZE = 32
=======
BATCH_SIZE = 40
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
        group["lr"] = 2.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    with torch.no_grad():
        if not hasattr(optimizer, "_ema_parameters"):
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            torch._foreach_lerp_(
                optimizer._ema_parameters,
                parameters,
                0.005,
            )
        if step + 1 >= total_steps:
            for parameter, average in zip(
                parameters, optimizer._ema_parameters
            ):
                parameter.copy_(average)
>>>>>>> REPLACE