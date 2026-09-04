MECHANISM: Late-trajectory exponential weight averaging

HYPOTHESIS: Applying a 0.995-decay parameter EMA to the best linear flip-loss curriculum will exceed 9,330 correct predictions by reducing late-training weight noise without altering exposure or inference cost.

INTENDED_EDIT: Restore linear ensemble-loss annealing and replace the final learned parameters with their optimizer-step EMA before validation.

EVIDENCE: Linear ensemble annealing achieved the best result at 9,330 correct, while cosine weighting fell to 9,322 and learning-rate-floor changes failed or regressed; this preserves the winning curriculum and tests an orthogonal stabilization mechanism.

<<<<<<< SEARCH
    progress = step / max(total_steps - 1, 1)
    ensemble_weight = 0.75 - 0.25 * math.cos(math.pi * progress)
=======
    ensemble_weight = 0.5 + 0.5 * step / max(total_steps - 1, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    peak_lr = 3.0e-3
=======
) -> None:
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
                average.lerp_(parameter.detach(), 0.005)

        if step >= total_steps - 1:
            for parameter, average in zip(parameters, ema_parameters):
                parameter.copy_(average)

    peak_lr = 3.0e-3
>>>>>>> REPLACE