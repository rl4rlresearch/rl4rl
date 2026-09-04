MECHANISM: Late-trajectory exponential weight averaging

HYPOTHESIS: EMA over the final quarter of training, combined with the best verified 0.04→0 smoothing schedule, will exceed 9,240 correct predictions by reducing sensitivity to individual late updates while retaining the proven nonzero learning-rate floor.

INTENDED_EDIT: Adopt the verified 1.296352 calibration and cosine 0.04→0 smoothing, then average learned parameters with decay 0.95 during the final quarter and install the averaged weights after the last optimizer step.

EVIDENCE: The 0.04→0 design achieved the best verified score with 9,240 correct, while zero and doubled terminal learning-rate floors fell to 9,219 and 9,205; late optimization is therefore consequential, motivating trajectory averaging without changing the successful floor.

<<<<<<< SEARCH
        return 1.30 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    completed_steps = getattr(optimizer, "_completed_steps", 0) + 1
    optimizer._completed_steps = completed_steps
    average_progress = min(completed_steps / max(total_steps, 1), 1.0)
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]

    with torch.no_grad():
        if average_progress >= 0.75:
            averaged = getattr(optimizer, "_averaged_parameters", None)
            if averaged is None:
                averaged = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._averaged_parameters = averaged
            else:
                for average, parameter in zip(averaged, parameters):
                    average.lerp_(parameter.detach(), 0.05)

        if completed_steps >= total_steps:
            averaged = getattr(optimizer, "_averaged_parameters", None)
            if averaged is not None:
                for parameter, average in zip(parameters, averaged):
                    parameter.copy_(average)
>>>>>>> REPLACE