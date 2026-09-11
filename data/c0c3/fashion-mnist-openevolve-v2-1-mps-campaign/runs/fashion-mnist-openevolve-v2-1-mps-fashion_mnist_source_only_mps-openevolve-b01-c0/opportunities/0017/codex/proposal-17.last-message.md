MECHANISM: Late-strengthened dropout curriculum

HYPOTHESIS: Ramping dense-head dropout from 5% to 15% while preserving its 10% average will exceed 9,290 correct predictions by easing early optimization without losing the regularization whose removal reduced accuracy to 9,270.

INTENDED_EDIT: Replace fixed 10% training dropout with a linear 5%→15% schedule over the fixed exposure.

EVIDENCE: Removing 10% dropout reduced validation correct from 9,290 to 9,270, showing that dropout is useful; scheduling the same average strength tests whether its early optimization cost can be reduced without sacrificing late regularization.

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
    model.classifier[4].p = 0.05 + 0.10 * progress
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
>>>>>>> REPLACE