MECHANISM: Exponential moving-average checkpoint ensembling

HYPOTHESIS: Evaluating a 0.99-decay exponential average of the optimization trajectory will exceed 9,247 correct predictions by reducing endpoint variance without changing examples, augmentation, architecture, or training time materially.

INTENDED_EDIT: Maintain an exponential moving average of every learned parameter after each optimizer step and copy the averaged parameters into the model after the final step.

EVIDENCE: Loss-schedule refinements and added architectural features tied or regressed from the 9,247-correct design, while parameter averaging remains an untested temporal-ensemble axis that preserves the proven training objective.

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
    optimizer._ema_pairs = [
        (parameter, parameter.detach().clone())
        for parameter in model.parameters()
    ]
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    warmup_fraction = 0.05
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    with torch.no_grad():
        for parameter, average in optimizer._ema_pairs:
            average.lerp_(parameter, 0.01)
        if step + 1 >= total_steps:
            for parameter, average in optimizer._ema_pairs:
                parameter.copy_(average)
    warmup_fraction = 0.05
>>>>>>> REPLACE