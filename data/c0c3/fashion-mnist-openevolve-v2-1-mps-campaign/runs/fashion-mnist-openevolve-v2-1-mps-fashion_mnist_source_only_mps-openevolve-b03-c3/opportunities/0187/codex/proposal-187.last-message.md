MECHANISM: Late-stage parameter-and-BatchNorm exponential moving average

HYPOTHESIS: Averaging the final quarter of the cosine training trajectory will exceed 9,284 correct predictions by reducing final-iterate noise while retaining the validated architecture, augmentation, and logit-space TTA.

INTENDED_EDIT: Restore the best validated 1.16727 calibration and maintain a decay-0.95 EMA of parameters and floating BatchNorm buffers during the final training quarter, installing the averaged state after the last optimizer step.

EVIDENCE: Calibration refinements saturated at 9,284 correct, while channel attention, translation-distribution changes, and probability-space TTA reduced accuracy; this motivates preserving the established computation and stabilizing only its late learned trajectory.

<<<<<<< SEARCH
        return 1.1682 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
    optimizer._ema_calls = 0
    optimizer._ema_parameters = None
    optimizer._ema_model_buffers = [
        buffer
        for buffer in model.buffers()
        if torch.is_floating_point(buffer)
    ]
    optimizer._ema_buffers = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    optimizer._ema_calls += 1
    if optimizer._ema_calls >= math.ceil(0.75 * total_steps):
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            if optimizer._ema_parameters is None:
                optimizer._ema_parameters = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer._ema_buffers = [
                    buffer.detach().clone()
                    for buffer in optimizer._ema_model_buffers
                ]
            else:
                for average, parameter in zip(
                    optimizer._ema_parameters, parameters
                ):
                    average.lerp_(parameter.detach(), 0.05)
                for average, buffer in zip(
                    optimizer._ema_buffers, optimizer._ema_model_buffers
                ):
                    average.lerp_(buffer.detach(), 0.05)

            if optimizer._ema_calls >= total_steps:
                for parameter, average in zip(
                    parameters, optimizer._ema_parameters
                ):
                    parameter.copy_(average)
                for buffer, average in zip(
                    optimizer._ema_model_buffers, optimizer._ema_buffers
                ):
                    buffer.copy_(average)
>>>>>>> REPLACE