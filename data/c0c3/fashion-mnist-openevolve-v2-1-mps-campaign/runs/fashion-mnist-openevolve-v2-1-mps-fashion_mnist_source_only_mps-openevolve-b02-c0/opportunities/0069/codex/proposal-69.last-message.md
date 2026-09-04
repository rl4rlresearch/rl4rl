MECHANISM: Late hard-label consolidation

HYPOTHESIS: Annealing label smoothing from 0.03 to zero over the final quarter of training will increase validation_correct above 9,252 by preserving early regularization while allowing low-learning-rate updates to sharpen class boundaries.

INTENDED_EDIT: Replace constant label smoothing with a step-aware linear decay during the final 25% of optimizer steps; architecture, augmentation, optimizer, averaging, and inference remain unchanged.

EVIDENCE: The verified model’s cross-entropy improved when inference logits were sharpened from scale 1.05 to 1.10 without changing predictions, indicating lingering underconfidence; this compute-neutral training-side change tests whether removing smoothing late can also improve argmax decisions without worsening the recurring runtime constraint.

<<<<<<< SEARCH
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
def training_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    total_steps: int,
) -> torch.Tensor:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    smoothing = 0.03
    if progress > 0.75:
        smoothing *= (1.0 - progress) / 0.25
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE