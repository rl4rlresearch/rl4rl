MECHANISM: Linearly annealed label smoothing

HYPOTHESIS: Annealing label smoothing from 0.04 to 0 while retaining approximately the baseline’s 0.02 average will exceed 9,290 correct predictions by regularizing early learning while allowing sharper class separation near training’s end.

INTENDED_EDIT: Replace constant 0.02 label smoothing with a linear 0.04-to-0 schedule across the fixed training exposure.

EVIDENCE: Reducing head dropout from 0.10 to 0.05 lowered validation correct from 9,290 to 9,275, indicating that regularization remains useful; scheduling smoothing preserves its overall strength while avoiding persistent late-stage target bias.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    label_smoothing = 0.04 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE