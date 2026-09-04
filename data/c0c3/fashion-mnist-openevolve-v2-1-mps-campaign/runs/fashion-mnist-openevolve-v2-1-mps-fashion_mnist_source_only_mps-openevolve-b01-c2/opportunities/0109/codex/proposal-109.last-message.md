MECHANISM: Sparse tail-iterate weight averaging

HYPOTHESIS: Averaging 17 evenly spaced parameter states from the final 128 low-learning-rate steps will exceed 9,330 correct predictions by reducing endpoint variance without the full-training overhead of the timed-out EMA attempt.

INTENDED_EDIT: Retain the verified architecture, objective, schedule, and 1.184 calibration, while sparsely averaging late parameter states and installing the average after the final optimizer step.

EVIDENCE: The current design verifies 9,330 correct, while full-training EMA did not finish within the time limit; restricting averaging to sparse samples from the cosine schedule’s low-rate tail tests the same variance-reduction idea with negligible added work.

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
    if step < warmup_steps:
        multiplier = 0.1 + 0.9 * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.02 + 0.98 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    peak_lr = 3.0e-3
    warmup_steps = max(1, int(0.05 * total_steps))
    if step < warmup_steps:
        multiplier = 0.1 + 0.9 * (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        multiplier = 0.02 + 0.98 * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = peak_lr * multiplier

    averaging_start = max(0, total_steps - 128)
    should_average = (
        step >= averaging_start
        and (
            (step - averaging_start) % 8 == 0
            or step == total_steps - 1
        )
    )
    if should_average:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            if not hasattr(optimizer, "_tail_average"):
                optimizer._tail_average = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._tail_average_count = 1
            else:
                optimizer._tail_average_count += 1
                weight = 1.0 / optimizer._tail_average_count
                for average, parameter in zip(
                    optimizer._tail_average,
                    parameters,
                ):
                    average.lerp_(parameter.detach(), weight)

            if step == total_steps - 1:
                for parameter, average in zip(
                    parameters,
                    optimizer._tail_average,
                ):
                    parameter.copy_(average)
>>>>>>> REPLACE