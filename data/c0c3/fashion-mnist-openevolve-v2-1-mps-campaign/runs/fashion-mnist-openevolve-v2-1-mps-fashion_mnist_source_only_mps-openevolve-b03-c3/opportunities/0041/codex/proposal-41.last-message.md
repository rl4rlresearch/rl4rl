MECHANISM: End-of-training exponential weight averaging

HYPOTHESIS: Replacing the final optimizer iterate with a 0.98-decay exponential average of the same model weights will exceed 9,240 correct predictions by reducing late-training parameter noise without disturbing the proven architecture, augmentation, or evaluation calibration.

INTENDED_EDIT: Track an EMA of all learned parameters after each optimizer step and install it after the final step; retain final BatchNorm statistics and the verified 1.296352 logit scale.

EVIDENCE: The calibrated baseline consistently reaches 9,240 correct, while adding a global head reduced this to 9,227 and coordinate gating to 9,225; preserving the successful representation and changing only final-weight selection is the clearest orthogonal next test.

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
    )
    optimizer._ema_parameters = tuple(model.parameters())
    optimizer._ema_shadow = None
    optimizer._ema_updates = 0
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    with torch.no_grad():
        parameters = optimizer._ema_parameters
        if optimizer._ema_shadow is None:
            optimizer._ema_shadow = [
                parameter.detach().clone() for parameter in parameters
            ]
        else:
            for shadow, parameter in zip(
                optimizer._ema_shadow, parameters
            ):
                shadow.lerp_(parameter.detach(), 0.02)

        optimizer._ema_updates += 1
        if optimizer._ema_updates >= total_steps:
            for parameter, shadow in zip(
                parameters, optimizer._ema_shadow
            ):
                parameter.copy_(shadow)
>>>>>>> REPLACE