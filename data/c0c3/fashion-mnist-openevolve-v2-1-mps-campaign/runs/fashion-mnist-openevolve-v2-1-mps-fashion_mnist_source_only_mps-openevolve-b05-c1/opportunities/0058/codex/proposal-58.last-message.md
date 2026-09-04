MECHANISM: Warm-started exponential weight averaging

HYPOTHESIS: Averaging the late optimization trajectory with a warm-started 0.99-decay EMA will exceed 9,334 correct predictions by reducing final-weight noise without adding learned parameters or expensive forward passes.

INTENDED_EDIT: Maintain an adaptive EMA of all learned parameters after optimizer steps and install the averaged weights after the final step.

EVIDENCE: The global-statistics bypass reached 9,334 correct, while additional feature computations repeatedly exceeded the verification time limit; parameter averaging preserves the proven architecture and adds negligible training cost.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LR * 0.2,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    optimizer.ema_pairs = None
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    with torch.no_grad():
        if optimizer.ema_pairs is None:
            optimizer.ema_pairs = [
                (parameter, parameter.detach().clone())
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
        else:
            updates = step + 1
            ema_decay = min(0.99, (updates + 1.0) / (updates + 10.0))
            for parameter, average in optimizer.ema_pairs:
                average.lerp_(parameter, 1.0 - ema_decay)

        if step + 1 >= total_steps:
            for parameter, average in optimizer.ema_pairs:
                parameter.copy_(average)

    progress = min((step + 1) / max(total_steps, 1), 1.0)
>>>>>>> REPLACE