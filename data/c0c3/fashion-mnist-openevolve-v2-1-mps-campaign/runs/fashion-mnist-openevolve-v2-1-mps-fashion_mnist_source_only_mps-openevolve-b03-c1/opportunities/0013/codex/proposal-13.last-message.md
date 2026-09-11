MECHANISM: Endpoint-anchored short-horizon weight averaging

HYPOTHESIS: Blending 25% of a bias-corrected 0.99 EMA into the final weights will exceed the current 9122.400048 validation score by preserving endpoint accuracy while inheriting some of the EMA’s lower cross-entropy.

INTENDED_EDIT: Track a parameter EMA after every optimizer step, then interpolate the final parameters 25% toward that EMA only at the last fixed-budget step.

EVIDENCE: Full EMA produced 9,121 correct versus the endpoint’s 9,122, but improved cross-entropy substantially from 0.24985 to 0.24546; this indicates useful trajectory smoothing whose full-strength replacement was slightly too aggressive.

<<<<<<< SEARCH
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
=======
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 4e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.25e-4,
        betas=(0.9, 0.99),
    )
    optimizer._ema_shadow = [
        torch.zeros_like(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer._ema_updates = 0
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
=======
) -> None:
    ema_decay = 0.99
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    with torch.no_grad():
        optimizer._ema_updates += 1
        for shadow, parameter in zip(optimizer._ema_shadow, parameters):
            shadow.mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
        if step + 1 >= total_steps:
            correction = 1.0 - ema_decay ** optimizer._ema_updates
            for shadow, parameter in zip(optimizer._ema_shadow, parameters):
                parameter.lerp_(shadow / correction, 0.25)

    progress = min((step + 1) / max(total_steps, 1), 1.0)
>>>>>>> REPLACE