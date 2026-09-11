MECHANISM: Iso-mean increasing label-smoothing schedule

HYPOTHESIS: Increasing label smoothing linearly from 0.010 to 0.020 will exceed 9,166 correct predictions by concentrating regularization late in training while preserving the baseline’s average smoothing strength.

INTENDED_EDIT: Replace constant 0.015 label smoothing with a linear 0.010-to-0.020 schedule over the fixed training exposure.

EVIDENCE: Annealing smoothing toward zero reduced correctness from 9,166 to 9,160, indicating that late-training smoothing is valuable; keeping the schedule’s mean at 0.015 avoids the stronger average regularization and distribution distortion associated with less successful alternatives.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.015)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    label_smoothing = 0.010 + 0.010 * progress
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE