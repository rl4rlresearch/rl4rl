MECHANISM: Tail exponential weight averaging

HYPOTHESIS: Averaging the successful model’s final half-training trajectory will exceed 9,168 correct predictions by reducing optimizer noise without introducing the harmful input transformations seen in flip ensembling.

INTENDED_EDIT: Maintain a 0.99-decay EMA of model parameters during the latter half of training and install the averaged weights after the final optimizer step.

EVIDENCE: The unmodified training path reached 9,168 correct, while horizontal-flip evaluation fell to 8,889; weight-space averaging preserves the validated input distribution while providing parameter-free ensembling.

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier

    first_step = getattr(optimizer, "_first_training_step", None)
    if first_step is None:
        first_step = step
        optimizer._first_training_step = step
    completed_steps = step - first_step + 1

    if completed_steps >= total_steps // 2:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        ema_parameters = getattr(optimizer, "_ema_parameters", None)
        with torch.no_grad():
            if ema_parameters is None:
                ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_parameters = ema_parameters
            else:
                for average, parameter in zip(ema_parameters, parameters):
                    average.lerp_(parameter.detach(), 0.01)

            if completed_steps >= total_steps:
                for parameter, average in zip(parameters, ema_parameters):
                    parameter.copy_(average)
>>>>>>> REPLACE