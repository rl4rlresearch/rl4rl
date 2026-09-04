MECHANISM: Early weak mixup with clean-tail fine-tuning

HYPOTHESIS: Applying 10% label-aware mixup during the first 75% of optimization, followed by clean low-learning-rate updates, will exceed 9,162 correct predictions by improving generalization without preventing final adaptation to unmixed validation images.

INTENDED_EDIT: Add fixed-strength batch mixup to the early training loss while preserving the existing label smoothing, architecture, parameter count, schedule, and final quarter of clean training.

EVIDENCE: Dual-statistic attention achieved 9,162 correct, while repeated attention refinements and an added edge-strength channel failed to improve; this motivates testing an orthogonal, computationally cheap loss-level regularizer.

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
    if step * 4 < total_steps * 3:
        permutation = torch.randperm(images.shape[0], device=images.device)
        mixed_images = 0.9 * images + 0.1 * images[permutation]
        logits = model(mixed_images)
        primary_loss = F.cross_entropy(
            logits, labels, label_smoothing=0.015
        )
        mixed_loss = F.cross_entropy(
            logits, labels[permutation], label_smoothing=0.015
        )
        return 0.9 * primary_loss + 0.1 * mixed_loss

    return F.cross_entropy(model(images), labels, label_smoothing=0.015)
>>>>>>> REPLACE