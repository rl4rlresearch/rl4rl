MECHANISM: Mean-preserving label-smoothing annealing

HYPOTHESIS: Annealing label smoothing from 0.04 to 0 while preserving the original average strength of 0.02 will exceed 9,285 correct predictions by regularizing early learning and sharpening class margins during the low-learning-rate tail.

INTENDED_EDIT: Replace constant 0.02 label smoothing with a linear exposure-progress schedule from approximately 0.04 to 0; architecture, augmentation, optimizer, and averaging cost remain unchanged.

EVIDENCE: Averaging-density variants moved correctness by only 2–4 images, plateauing at 9,285, while every verified design retained constant 0.02 smoothing. This isolates an untested, computation-neutral training lever suited to the accuracy-first objective.

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
    label_smoothing = 0.04 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE