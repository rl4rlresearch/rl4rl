MECHANISM: Cosine-decayed label smoothing with linear-recency tail averaging

HYPOTHESIS: Concentrating the same mean smoothing strength early and approaching hard labels during the averaged tail will exceed 9,287 correct predictions by regularizing high-learning-rate training while sharpening late decision boundaries.

INTENDED_EDIT: Restore the strongest linear-recency averaging baseline and replace constant 0.02 label smoothing with a cosine decay from 0.04 to 0.

EVIDENCE: Recency powers 0.5, 0.75, and 1.0 all plateaued at 9,287 correct, with linear recency producing the lowest cross-entropy; this motivates restoring that baseline and testing the previously unchanged loss schedule.

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
    progress = min((step + 1) / max(total_steps, 1), 1.0)
    label_smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        recency_power = 0.5
=======
        recency_power = 1.0
>>>>>>> REPLACE