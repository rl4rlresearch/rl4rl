MECHANISM: Sparse exponential weight averaging

HYPOTHESIS: Averaging the strongest verified top-four model’s late optimization trajectory will achieve at least 9,323 correct predictions by reducing checkpoint noise without altering its architecture or training objective.

INTENDED_EDIT: Maintain an exponential moving average of learned parameters, update it every 16 optimizer steps with decay 0.90, and install the averaged weights after the final step.

EVIDENCE: Top-four attention produced the best result at 9,322 correct, while attention, augmentation, and consistency-loss changes generally regressed or timed out; weight averaging preserves that implementation and tests an orthogonal, low-overhead generalization refinement.

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
    optimizer.ema_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    if step % 16 == 0 or step >= total_steps:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            for average, parameter in zip(
                optimizer.ema_parameters, parameters
            ):
                average.lerp_(parameter, 0.10)
            if step >= total_steps:
                for parameter, average in zip(
                    parameters, optimizer.ema_parameters
                ):
                    parameter.copy_(average)

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE