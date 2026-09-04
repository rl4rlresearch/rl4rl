MECHANISM: Late-training exponential weight averaging

HYPOTHESIS: Averaging the final half of the proven small-batch optimization trajectory will exceed 9,215 correct predictions by reducing parameter and BatchNorm-state variance without adding learned parameters.

INTENDED_EDIT: Track an EMA of the complete model state after 50% of training and install the averaged state after the final optimizer step.

EVIDENCE: Halving the batch size improved the 216,250-parameter model from 9,212 to 9,215 correct while the larger model regressed, indicating that further gains are more likely from stabilizing optimization than increasing capacity.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.1e-3,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.1e-3,
        betas=(0.9, 0.99),
        weight_decay=2e-4,
    )
    optimizer.ema_model = model
    optimizer.ema_state = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    decay_progress = max(progress - 0.10, 0.0) / 0.90
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    decay_progress = max(progress - 0.10, 0.0) / 0.90
    multiplier = 0.10 + 0.90 * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier

    if progress >= 0.50:
        current_state = optimizer.ema_model.state_dict()
        with torch.no_grad():
            if optimizer.ema_state is None:
                optimizer.ema_state = {
                    name: value.detach().clone()
                    for name, value in current_state.items()
                }
            else:
                for name, value in current_state.items():
                    averaged = optimizer.ema_state[name]
                    if torch.is_floating_point(averaged):
                        averaged.lerp_(value.detach(), 0.01)
                    else:
                        averaged.copy_(value)

            if step >= total_steps:
                for name, value in current_state.items():
                    value.copy_(optimizer.ema_state[name])
>>>>>>> REPLACE