MECHANISM: Bias-corrected exponential weight averaging

HYPOTHESIS: Evaluating a recent-history exponential average of the unchanged model will exceed 9,257 correct predictions by reducing optimizer noise without weakening the proven training objective or architecture.

INTENDED_EDIT: Track EMA copies of learned parameters and floating-point normalization buffers after every optimizer step, then install the averaged state after the final step.

EVIDENCE: The 233,434-parameter design remains best, while architectural changes and decayed smoothing regressed; weight averaging preserves that design and its successful fixed-smoothing trajectory while targeting generalization variance directly.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=2.0e-4,
        betas=(0.9, 0.99),
    )
    optimizer._ema_model = model
    optimizer._ema_buffers = [
        buffer.detach().clone() if buffer.is_floating_point() else None
        for buffer in model.buffers()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 3.0e-3 * multiplier

    ema_decay = min(0.99, (step + 1) / (step + 10))
    with torch.no_grad():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                state = optimizer.state[parameter]
                parameter_ema = state.get("model_ema")
                if parameter_ema is None:
                    parameter_ema = parameter.detach().clone()
                    state["model_ema"] = parameter_ema
                else:
                    parameter_ema.mul_(ema_decay).add_(
                        parameter, alpha=1.0 - ema_decay
                    )

        model = optimizer._ema_model
        for buffer, buffer_ema in zip(
            model.buffers(), optimizer._ema_buffers
        ):
            if buffer_ema is not None:
                buffer_ema.mul_(ema_decay).add_(
                    buffer, alpha=1.0 - ema_decay
                )

        if step + 1 >= total_steps:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.copy_(
                        optimizer.state[parameter]["model_ema"]
                    )
            for buffer, buffer_ema in zip(
                model.buffers(), optimizer._ema_buffers
            ):
                if buffer_ema is not None:
                    buffer.copy_(buffer_ema)
>>>>>>> REPLACE