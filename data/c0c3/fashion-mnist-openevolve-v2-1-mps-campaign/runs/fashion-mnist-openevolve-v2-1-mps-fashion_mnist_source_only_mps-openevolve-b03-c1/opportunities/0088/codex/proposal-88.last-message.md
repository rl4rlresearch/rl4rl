MECHANISM: Late-stage label-smoothing annealing

HYPOTHESIS: Annealing label smoothing from 0.015 to zero during the final quarter will exceed 9,166 correct predictions by retaining early regularization while sharpening decision boundaries before validation.

INTENDED_EDIT: Preserve the existing loss for the first 75% of training, then linearly decay label smoothing to zero over the remaining optimizer steps.

EVIDENCE: Early mixup reduced correctness from 9,166 to 9,130, suggesting excess soft-target regularization is harmful, while reduced dropout reached only 9,141; this motivates removing target smoothing only late while preserving the successful structural regularization.

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
    progress = (step + 1) / max(total_steps, 1)
    smoothing_multiplier = min(1.0, max(0.0, 4.0 * (1.0 - progress)))
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=0.015 * smoothing_multiplier,
    )
>>>>>>> REPLACE